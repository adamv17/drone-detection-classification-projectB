clear; clc; close all;

% --- CONFIGURATION ---
dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 
holdoutRatio = 0.2; % 20% of files for testing

% --- 1. FILE DISCOVERY & BINARY LABELING ---
wavFiles = dir(fullfile(dataPath, '*.wav'));
numFiles = length(wavFiles);

fileLabels = cell(numFiles, 1);

for i = 1:numFiles
    nameParts = split(wavFiles(i).name, '_');
    rawLabel = nameParts{1};
    
    % --- BINARY MAPPING LOGIC ---
    % Check if label is 'Drone' (case-insensitive)
    if strcmpi(rawLabel, 'Drone')
        fileLabels{i} = 'DRONE';
    else
        fileLabels{i} = 'OTHER';
    end
end

% --- 2. STRATIFIED SPLIT ---
% Now splitting based on the NEW binary labels
cv = cvpartition(fileLabels, 'HoldOut', holdoutRatio);
isTrainFile = training(cv); 
isTestFile  = test(cv);

% --- 3. VERIFY BALANCE (PRINT STATS) ---
disp('------------------------------------------------');
disp('VERIFYING BINARY SPLIT BALANCE');
disp('------------------------------------------------');
uniqueLabels = unique(fileLabels);
fprintf('%-15s | %-10s | %-10s\n', 'Class', 'Train Qty', 'Test Qty');
fprintf('%-15s | %-10s | %-10s\n', '-----', '---------', '--------');

for i = 1:length(uniqueLabels)
    lbl = uniqueLabels{i};
    isThisClass = strcmp(fileLabels, lbl);
    
    nTrain = sum(isThisClass & isTrainFile);
    nTest  = sum(isThisClass & isTestFile);
    
    fprintf('%-15s | %-10d | %-10d\n', lbl, nTrain, nTest);
end
disp('------------------------------------------------');

% --- 4. EXTRACTION LOOP ---
featResults = cell(numFiles, 1);
labelResults = cell(numFiles, 1);
setResults   = cell(numFiles, 1); 

disp('Extracting features...');

parfor i = 1:numFiles
    try
        % Determine Set
        currentSet = "Test";
        if isTrainFile(i), currentSet = "Train"; end
        
        % Load Audio
        fileEntry = wavFiles(i);
        [audioData, fs] = audioread(fullfile(dataPath, fileEntry.name));
        if size(audioData, 2) > 1, audioData = mean(audioData, 2); end
        
        % MFCC
        winLen = round(0.03 * fs); 
        overlap = round(0.02 * fs); 
        aFE = audioFeatureExtractor('SampleRate', fs, ...
            'Window', hamming(winLen, 'periodic'), ... 
            'OverlapLength', overlap, ...              
            'mfcc', true);
        
        features = extract(aFE, audioData);
        
        % --- RE-APPLY BINARY LABELING ---
        nameParts = split(fileEntry.name, '_');
        rawLabel = nameParts{1};
        if strcmpi(rawLabel, 'Drone')
            finalLabel = 'DRONE';
        else
            finalLabel = 'OTHER';
        end
        
        numFrames = size(features, 1);
        
        % Store
        featResults{i} = features;
        labelResults{i} = repmat({finalLabel}, numFrames, 1);
        setResults{i}   = repmat(currentSet, numFrames, 1); 
        
    catch
        continue;
    end
end

% --- 5. FINALIZE TABLES ---
disp('Building final tables...');
emptyIdx = cellfun(@isempty, featResults);
X_All = vertcat(featResults{~emptyIdx});
Y_All = vertcat(labelResults{~emptyIdx});
S_All = vertcat(setResults{~emptyIdx});

% Create Table
numFeats = size(X_All, 2);
varNames = arrayfun(@(x) sprintf('MFCC_%d', x), 1:numFeats, 'UniformOutput', false);
FullTable = array2table(X_All, 'VariableNames', varNames);
FullTable.Label = categorical(Y_All);

% Split
TrainTable = FullTable(S_All == "Train", :);
TestTable  = FullTable(S_All == "Test", :);

disp('Done! "TrainTable" and "TestTable" are ready for Classification Learner.');