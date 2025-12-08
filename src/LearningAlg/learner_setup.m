%% learner_setup.m
clear; clc; close all;

% --- 1. CONFIGURATION ---
dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 
holdoutRatio = 0.2; 

% PART A: Standard AudioFeatureExtractor Settings
afeConfig.mfcc = true;              
afeConfig.spectralCentroid = true;  
afeConfig.spectralRolloffPoint = true; 
afeConfig.spectralFlux = true;      
afeConfig.zerocrossrate = true;  
afeConfig.pitch = false;            

% PART B: Custom Feature Settings
customConfig.bicoherence = true; 

% --- 2. FILE DISCOVERY & LABELING ---
wavFiles = dir(fullfile(dataPath, '*.wav'));
if isempty(wavFiles)
    error('No .wav files found in %s', dataPath);
end

numFiles = length(wavFiles);
fileLabels = cell(numFiles, 1);

fprintf('Found %d files. Assigning Labels...\n', numFiles);
for i = 1:numFiles
    nameParts = split(wavFiles(i).name, '_');
    if strcmpi(nameParts{1}, 'Drone')
        fileLabels{i} = 'DRONE';
    else
        fileLabels{i} = 'OTHER';
    end
end

% --- 3. REPRODUCIBLE PARTITIONING ---
rng(42); 
cv = cvpartition(fileLabels, 'HoldOut', holdoutRatio);
isTrainFile = training(cv); 
isTestFile  = test(cv);

fprintf('\nData Partition Summary:\n');
uniqueLabels = unique(fileLabels);
for i = 1:length(uniqueLabels)
    lbl = uniqueLabels{i};
    isThisClass = strcmp(fileLabels, lbl);
    fprintf('  %-10s: Train=%d, Test=%d\n', lbl, sum(isThisClass & isTrainFile), sum(isThisClass & isTestFile));
end

% --- 4. FEATURE EXTRACTION LOOP ---
featResults = cell(numFiles, 1);
labelResults = cell(numFiles, 1);
setResults   = cell(numFiles, 1); 
capturedInfo = []; 

activeStandard = struct2cell(afeConfig);
hasStandardFeatures = any([activeStandard{:}]);

% Define Window Sizes
% Standard (Fast)
winLen_Std = round(0.03 * 44100); % 30ms
overlap_Std = round(0.02 * 44100); % 20ms (10ms hop)
hopSize = winLen_Std - overlap_Std;

% Custom (Slow/Statistical)
winLen_Custom = round(0.50 * 44100); % 500ms (Half-second context)

fprintf('\nExtracting Features...\n');

for i = 1:numFiles
    try
        currentSet = "Test";
        if isTrainFile(i), currentSet = "Train"; end
        
        fileEntry = wavFiles(i);
        [audioData, fs] = audioread(fullfile(dataPath, fileEntry.name));
        if size(audioData, 2) > 1, audioData = audioData(:,1); end
        
        % --- C. STANDARD FEATURES ---
        afe_features = [];
        numFrames = 0;
        
        if hasStandardFeatures
            % Recalculate windows based on actual fs of file
            winLen = round(0.03 * fs);       
            overlap = round(0.02 * fs);      
            hop = winLen - overlap;
            
            aFE = audioFeatureExtractor('SampleRate', fs, ...
                'Window', hamming(winLen, 'periodic'), ... 
                'OverlapLength', overlap);
            set(aFE, afeConfig); 
            
            afe_features = extract(aFE, audioData);
            
            if isempty(capturedInfo)
                capturedInfo = info(aFE);
            end
            numFrames = size(afe_features, 1);
        else
            % Fallback frame counting if no standard features
            hop = round(0.01 * fs); % 10ms hop
            numFrames = floor((length(audioData) - winLen_Custom) / hop);
            afe_features = zeros(numFrames, 0); 
        end
        
        % --- D. CUSTOM FEATURES (OPTIMIZED: SAMPLE & HOLD) ---
        custom_features = [];
        
        if customConfig.bicoherence
            % 1. Setup Timing
            % Calculate Bicoherence every 0.25 seconds (4 times/sec)
            bico_step_time = 0.25; 
            bico_step_samples = round(bico_step_time * fs);
            
            % The window size for calculation remains large (0.5s) for statistics
            longWin = round(0.50 * fs); 
            
            % 2. Pre-allocate
            % Run once on zeros to get the feature width (11 features)
            dummyFeat = getBicoherenceFeature(zeros(longWin, 1), fs);
            numCustomBins = length(dummyFeat);
            
            % We will fill this matrix row-by-row
            this_file_custom = zeros(numFrames, numCustomBins);
            
            % 3. The Optimized Loop
            % Instead of k = 1:numFrames, we jump by 'bico_step_samples'
            % We map MFCC Frame Indices to Audio Sample Indices
            
            last_calc_feat = zeros(1, numCustomBins); % Store last known value
            
            for k = 1:numFrames
                % Convert MFCC Frame Index (k) to Audio Sample Index
                % Center of current MFCC frame
                currentCenter = round((k-1)*hop + (winLen/2));
                
                % Check if it's time to update the Bicoherence (every 0.25s)
                % Or if it's the very first frame
                if k == 1 || mod(currentCenter, bico_step_samples) < hop
                    
                    % Define the Large Window (0.5s) centered on this point
                    startIdx = currentCenter - floor(longWin/2);
                    endIdx   = startIdx + longWin - 1;
                    
                    % Extract Chunk with Padding
                    if startIdx < 1
                        chunk = [zeros(1-startIdx, 1); audioData(1:endIdx)];
                    elseif endIdx > length(audioData)
                        chunk = [audioData(startIdx:end); zeros(endIdx-length(audioData), 1)];
                    else
                        chunk = audioData(startIdx:endIdx);
                    end
                    
                    % --- HEAVY CALCULATION (Happens rarely) ---
                    last_calc_feat = getBicoherenceFeature(chunk, fs);
                end
                
                % --- LIGHT ASSIGNMENT (Happens every frame) ---
                % Just copy the last calculated value
                this_file_custom(k, :) = last_calc_feat;
            end
            
            custom_features = [custom_features, this_file_custom];
        end
        
        % --- E. STITCH TOGETHER ---
        final_features = [afe_features, custom_features];
        
        nameParts = split(fileEntry.name, '_');
        if strcmpi(nameParts{1}, 'Drone')
            finalLabel = 'DRONE';
        else
            finalLabel = 'OTHER';
        end
        
        if numFrames > 0
            featResults{i} = final_features;
            labelResults{i} = repmat({finalLabel}, numFrames, 1);
            setResults{i}   = repmat(currentSet, numFrames, 1); 
        end
        
    catch ME
        fprintf('Error processing %s: %s\n', wavFiles(i).name, ME.message);
        continue;
    end
end

% --- 5. ROBUST NAME GENERATION ---
varNames = {};

% Part A: Standard Names
if hasStandardFeatures && ~isempty(capturedInfo)
    extractorFields = fieldnames(capturedInfo); 
    for i = 1:length(extractorFields)
        featName = extractorFields{i};
        featIdxs = capturedInfo.(featName);
        featWidth = numel(featIdxs); 
        if featWidth == 1
            varNames{end+1} = featName; 
        else
            for k = 1:featWidth
                varNames{end+1} = sprintf('%s_%d', featName, k);
            end
        end
    end
end

% Part B: Custom Names
if customConfig.bicoherence
    varNames{end+1} = 'Bic_Max_All';
    varNames{end+1} = 'Bic_Max_Low';
    varNames{end+1} = 'Bic_Max_High';
    varNames{end+1} = 'Bic_SumSig_All';
    varNames{end+1} = 'Bic_SumSig_Low';
    varNames{end+1} = 'Bic_SumSig_High';
    varNames{end+1} = 'AIB_Rotor_0_300Hz';
    varNames{end+1} = 'AIB_Harmonic_300_1k';
    varNames{end+1} = 'AIB_Mid_1k_5k';
    varNames{end+1} = 'AIB_Empty_5k_10k';
    varNames{end+1} = 'AIB_PWM_10k_Plus';
end

if isempty(varNames)
    error('No features were enabled.');
end

% --- 6. BUILD TABLES ---
fprintf('\nConstructing Final Tables...\n');
emptyIdx = cellfun(@isempty, featResults);

if all(emptyIdx)
    error('No features extracted.');
end

X_All = vertcat(featResults{~emptyIdx});
Y_All = vertcat(labelResults{~emptyIdx});
S_All = vertcat(setResults{~emptyIdx});

if size(X_All, 2) ~= length(varNames)
    warning('Dimension mismatch: Data has %d cols, Names has %d.', size(X_All, 2), length(varNames));
    varNames = arrayfun(@(x) sprintf('Feat_%d', x), 1:size(X_All, 2), 'UniformOutput', false);
end

FullTable = array2table(X_All, 'VariableNames', varNames);
FullTable.Label = categorical(string(Y_All));

TrainTable = FullTable(S_All == "Train", :);
TestTable  = FullTable(S_All == "Test", :);

fprintf('DONE!\n');
fprintf('  TrainTable: %d rows\n', height(TrainTable));
fprintf('  TestTable:  %d rows\n', height(TestTable));
clearvars -except TrainTable TestTable afeConfig customConfig;
