%% learner_setup.m
clear; clc; close all;

% --- 1. CONFIGURATION ---
dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 
holdoutRatio = 0.2; 

% PART A: Standard AudioFeatureExtractor Settings (ONLY Built-in features)
afeConfig.mfcc = true;              
afeConfig.spectralCentroid = true;  
afeConfig.spectralRolloffPoint = true; 
afeConfig.spectralFlux = true;      
afeConfig.zerocrossrate = true;  
afeConfig.pitch = false;            

% PART B: Custom Feature Settings
customConfig.bicoherence = false; 

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

% --- 4. FEATURE EXTRACTION LOOP (ROBUST) ---
featResults = cell(numFiles, 1);
labelResults = cell(numFiles, 1);
setResults   = cell(numFiles, 1); 
capturedInfo = []; 

% Check if ANY standard features are enabled
activeStandard = struct2cell(afeConfig);
hasStandardFeatures = any([activeStandard{:}]);

fprintf('\nExtracting Features...\n');

for i = 1:numFiles
    try
        currentSet = "Test";
        if isTrainFile(i), currentSet = "Train"; end
        
        fileEntry = wavFiles(i);
        [audioData, fs] = audioread(fullfile(dataPath, fileEntry.name));
        if size(audioData, 2) > 1, audioData = audioData(:,1); end
        
        % --- A. CONFIGURE WINDOWING ---
        winLen = round(0.03 * fs);       
        overlap = round(0.02 * fs);      
        
        % --- B. MASTER CLOCK & BUFFERING ---
        % We ALWAYS buffer the audio to ensure we have raw frames available
        audioBuffered = buffer(audioData, winLen, overlap, 'nodelay');
        
        % --- C. STANDARD FEATURES ---
        afe_features = [];
        
        if hasStandardFeatures
            aFE = audioFeatureExtractor('SampleRate', fs, ...
                'Window', hamming(winLen, 'periodic'), ... 
                'OverlapLength', overlap);
            set(aFE, afeConfig); 
            
            afe_features = extract(aFE, audioData);
            
            % Capture info for naming (only needs to happen once)
            if isempty(capturedInfo)
                capturedInfo = info(aFE);
            end
            
            % If standard features exist, they dictate the Frame Count
            % (because 'extract' might drop the last incomplete frame)
            numFrames = size(afe_features, 1);
        else
            % Fallback: If no standard features, we use the buffer count
            numFrames = size(audioBuffered, 2);
            afe_features = zeros(numFrames, 0); % Empty matrix with correct rows
        end
        
        % --- D. CUSTOM FEATURES ---
        custom_features = [];
        
        if customConfig.bicoherence
            % Pre-allocate custom matrix
            % Run logic on first frame to detect width
            test_feat = getBicoherenceFeature(audioBuffered(:,1), fs); 
            numCustomBins = length(test_feat);
            customFeatWidth = numCustomBins; 
            
            this_file_custom = zeros(numFrames, numCustomBins);
            
            for k = 1:numFrames
                % Safety check: Ensure we don't exceed buffer dimensions
                % (Handles case where 'extract' dropped a frame but 'buffer' didn't)
                if k <= size(audioBuffered, 2)
                    frame = audioBuffered(:, k);
                    this_file_custom(k, :) = getBicoherenceFeature(frame, fs);
                else
                    this_file_custom(k, :) = zeros(1, numCustomBins);
                end
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
        
        % Only save if we actually got frames
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

% Part A: Standard Names (Only if we used them)
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
    % Use 'customFeatWidth' captured from the loop
    if customFeatWidth == 0
        % Fallback if loop failed to capture width (e.g. no files processed)
        % We simulate one run to get the width
        dummyFrame = zeros(round(0.03*44100), 1);
        dummyFeat = getBicoherenceFeature(dummyFrame, 44100);
        customFeatWidth = length(dummyFeat);
    end
    
    for k = 1:customFeatWidth
        varNames{end+1} = sprintf('Bicoherence_%d', k);
    end
end

% Check if we ended up with ANY names
if isempty(varNames)
    error('No features (Standard or Custom) were enabled/extracted.');
end

% --- 5. ROBUST NAME GENERATION ---
if isempty(capturedInfo)
    error('Feature extraction failed.');
end

varNames = {};

% Part A: Standard Names (From afeConfig)
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

% Part B: Custom Names (From customConfig)
if customConfig.bicoherence
    for k = 1:customFeatWidth
        varNames{end+1} = sprintf('Bicoherence_%d', k);
    end
end

% --- 6. BUILD TABLES ---
fprintf('\nConstructing Final Tables...\n');
emptyIdx = cellfun(@isempty, featResults);
X_All = vertcat(featResults{~emptyIdx});
Y_All = vertcat(labelResults{~emptyIdx});
S_All = vertcat(setResults{~emptyIdx});

if size(X_All, 2) ~= length(varNames)
    warning('Dimension mismatch: Data has %d cols, Names has %d.', size(X_All, 2), length(varNames));
    varNames = arrayfun(@(x) sprintf('Feat_%d', x), 1:size(X_All, 2), 'UniformOutput', false);
end

FullTable = array2table(X_All, 'VariableNames', varNames);
FullTable.Label = categorical(Y_All);

TrainTable = FullTable(S_All == "Train", :);
TestTable  = FullTable(S_All == "Test", :);

fprintf('DONE!\n');
fprintf('  TrainTable: %d rows\n', height(TrainTable));
fprintf('  TestTable:  %d rows\n', height(TestTable));
clearvars -except TrainTable TestTable afeConfig customConfig;

%% ---------------------------------------------------------
%  LOCAL FUNCTIONS
% ---------------------------------------------------------

function feat = getBicoherenceFeature(x, fs)
    % Placeholder for Bicoherence Logic
    % Currently returns 5 random numbers
    feat = rand(1, 5); 
end