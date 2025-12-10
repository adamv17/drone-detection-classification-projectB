%% learner_setup.m
clear; clc; close all;

% --- 1. CONFIGURATION ---
dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 
holdoutRatio = 0.2; 

% AFE Settings
afeConfig.mfcc = true;              
afeConfig.spectralCentroid = false;  
afeConfig.spectralRolloffPoint = false; 
afeConfig.spectralFlux = false;      
afeConfig.zerocrossrate = false;  
afeConfig.pitch = false;            

% Custom Settings
customConfig.bicoherence = false; 
customConfig.tkeo = true;

% --- 2. FILE DISCOVERY & LABELING ---
wavFiles = dir(fullfile(dataPath, '*.wav'));
if isempty(wavFiles)
    error('No .wav files found in %s', dataPath);
end

numFiles = length(wavFiles);
fileLabels = cell(numFiles, 1);

% --- 3. REPRODUCIBLE PARTITIONING ---
rng(42); 
for i = 1:numFiles
    nameParts = split(wavFiles(i).name, '_');
    if strcmpi(nameParts{1}, 'Drone')
        fileLabels{i} = 'DRONE';
    else
        fileLabels{i} = 'OTHER';
    end
end

cv = cvpartition(fileLabels, 'HoldOut', holdoutRatio);
isTrainFile = training(cv); 
isTestFile  = test(cv);

% --- 4. PRE-FLIGHT CHECK (GENERATE NAMES BEFORE LOOP) ---
% We run a dummy extraction to determine feature names/sizes 
% BEFORE entering the parallel loop.
fprintf('\nPre-calculating feature names...\n');

varNames = {};
activeStandard = struct2cell(afeConfig);
hasStandardFeatures = any([activeStandard{:}]);

% A. Standard Names (Calculated from Dummy AFE)
if hasStandardFeatures
    % Create a dummy extractor just to get the 'info' struct
    dummyFs = 44100;
    dummyWin = round(0.03 * dummyFs);
    dummyOv = round(0.02 * dummyFs);
    
    dummyAFE = audioFeatureExtractor('SampleRate', dummyFs, ...
        'Window', hamming(dummyWin), 'OverlapLength', dummyOv);
    set(dummyAFE, afeConfig);
    
    capturedInfo = info(dummyAFE);
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

% B. Custom Names (Hardcoded to match getBicoherenceFeature)
if customConfig.bicoherence
    dummyW = 6; % Known width of Bicoherence features
    % varNames{end+1} = 'Bic_Max_All';
    % varNames{end+1} = 'Bic_Max_Low';
    % varNames{end+1} = 'Bic_Max_High';
    % varNames{end+1} = 'Bic_SumSig_All';
    % varNames{end+1} = 'Bic_SumSig_Low';
    % varNames{end+1} = 'Bic_SumSig_High';
    varNames{end+1} = 'AIB_Heli_0_100Hz';
    varNames{end+1} = 'AIB_Drone_100_300Hz';
    varNames{end+1} = 'AIB_Harmonic_300_1k';
    varNames{end+1} = 'AIB_Mid_1k_5k';
    varNames{end+1} = 'AIB_Empty_5k_10k';
    varNames{end+1} = 'AIB_PWM_10k_Plus';
end
if customConfig.tkeo
    varNames{end+1} = 'TKEO_Mean';
    varNames{end+1} = 'TKEO_Std';
    varNames{end+1} = 'TKEO_Max';
    varNames{end+1} = 'TKEO_Kurtosis';
end

fprintf('Expected Features: %d\n', length(varNames));

% --- 5. FEATURE EXTRACTION LOOP ---
featResults = cell(numFiles, 1);
labelResults = cell(numFiles, 1);
setResults   = cell(numFiles, 1); 
nameResults  = cell(numFiles, 1); 

fprintf('\nExtracting Features (Parallel)...\n');

parfor i = 1:numFiles
    try
        currentSet = "Test";
        if isTrainFile(i), currentSet = "Train"; end
        
        fileEntry = wavFiles(i);
        fullPath = fullfile(dataPath, fileEntry.name);
        [audioData, fs] = audioread(fullPath);
        if size(audioData, 2) > 1, audioData = audioData(:, 1); end
        
        % A. Windowing
        winLen = round(0.03 * fs);       
        overlap = round(0.02 * fs);      
        hop = winLen - overlap;
        
        % B. Standard Features
        afe_features = [];
        numFrames = 0;
        
        if hasStandardFeatures
            aFE = audioFeatureExtractor('SampleRate', fs, ...
                'Window', hamming(winLen, 'periodic'), ... 
                'OverlapLength', overlap);
            set(aFE, afeConfig); 
            afe_features = extract(aFE, audioData);
            numFrames = size(afe_features, 1);
        else
            % Fallback
            longWin = round(0.50 * fs);
            numFrames = floor((length(audioData) - longWin) / hop);
            afe_features = zeros(numFrames, 0);
        end
        
        % C. Custom Features
        custom_features = [];
        
        if customConfig.bicoherence
            bico_step_time = 0.25; 
            bico_step_samples = round(bico_step_time * fs);
            longWin = round(0.50 * fs);
            
            this_file_custom = zeros(numFrames, dummyW);
            last_calc_feat = zeros(1, dummyW);
            
            for k = 1:numFrames
                currentCenter = round((k-1)*hop + (winLen/2));
                
                if k == 1 || mod(currentCenter, bico_step_samples) < hop
                    startIdx = currentCenter - floor(longWin/2);
                    endIdx   = startIdx + longWin - 1;
                    
                    if startIdx < 1
                        chunk = [zeros(1-startIdx, 1); audioData(1:endIdx)];
                    elseif endIdx > length(audioData)
                        chunk = [audioData(startIdx:end); zeros(endIdx-length(audioData), 1)];
                    else
                        chunk = audioData(startIdx:endIdx);
                    end
                    
                    last_calc_feat = getBicoherenceFeature(chunk, fs);
                end
                this_file_custom(k, :) = last_calc_feat;
            end
            custom_features = [custom_features, this_file_custom];
        end
        
        if customConfig.tkeo
            % We calculate TKEO on the Standard (Short) Frames
            % because impulses are short-lived events.
            
            % If you have standard features, use 'numFrames' from there.
            % If not, calculate numFrames from the buffer size.
            
            tkeo_width = 4; % Mean, Std, Max, Kurtosis
            this_file_tkeo = zeros(numFrames, tkeo_width);
            
            for k = 1:numFrames
                % Map frame index to sample center
                currentCenter = round((k-1)*hop + (winLen/2));
                
                % Extract short window (standard 30ms is fine for TKEO)
                sIdx = currentCenter - floor(winLen/2);
                eIdx = sIdx + winLen - 1;
                
                % Safe Extraction with Padding
                if sIdx < 1
                    chunk = [zeros(1-sIdx, 1); audioData(1:eIdx)];
                elseif eIdx > length(audioData)
                    chunk = [audioData(sIdx:end); zeros(eIdx-length(audioData), 1)];
                else
                    chunk = audioData(sIdx:eIdx);
                end
                
                % Calculate
                this_file_tkeo(k, :) = getNormTKEOFeatures(chunk);
            end
            custom_features = [custom_features, this_file_tkeo];
        end
        
        % D. Stitch & Store
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
            nameResults{i}  = repmat({fileEntry.name}, numFrames, 1);
        end
        
    catch ME
        fprintf('Error processing %s: %s\n', fileEntry.name, ME.message);
        continue;
    end
end

% --- 6. BUILD TABLES ---
emptyIdx = cellfun(@isempty, featResults);
X_All = vertcat(featResults{~emptyIdx});
Y_All = vertcat(labelResults{~emptyIdx});
S_All = vertcat(setResults{~emptyIdx});
N_All = vertcat(nameResults{~emptyIdx}); 

% Safety Check
if size(X_All, 2) ~= length(varNames)
    warning('Dimension Mismatch! Data: %d, Names: %d', size(X_All, 2), length(varNames));
    varNames = arrayfun(@(x) sprintf('Feat_%d', x), 1:size(X_All, 2), 'UniformOutput', false);
end

FullTable = array2table(X_All, 'VariableNames', varNames);
FullTable.Label = categorical(string(Y_All));
FullTable.Filename = string(N_All); 

TrainTable = FullTable(S_All == "Train", :);
TestTable  = FullTable(S_All == "Test", :);

fprintf('DONE. Tables contain "Filename" column.\n');
clearvars -except TrainTable TestTable afeConfig customConfig;