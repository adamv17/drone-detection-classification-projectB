clear; clc; close all;

% --- CONFIGURATION ---
dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio';
useMFCC = false;
useProny = true;
pronyOrder = 20; 

% Initialize lists for dynamic title
activeFeatureNames = {};
if useMFCC, activeFeatureNames{end+1} = 'MFCC'; end
if useProny, activeFeatureNames{end+1} = sprintf('Prony(Order=%d)', pronyOrder); end

wavFiles = dir(fullfile(dataPath, '*.wav'));
numFiles = length(wavFiles);

% --- PRE-ALLOCATE STORAGE (THE FIX) ---
% Use separate 1D arrays to avoid "multiple sliced access" errors
featResults = cell(numFiles, 1);
labelResults = cell(numFiles, 1);

disp(['Starting parallel extraction on ' num2str(numFiles) ' files...']);

parfor i = 1:numFiles
    % Re-construct file info inside the loop
    fileEntry = wavFiles(i);
    baseFileName = fileEntry.name;
    fullFileName = fullfile(dataPath, baseFileName);
    
    % Extract Label
    nameParts = split(baseFileName, '_');
    trueLabel = nameParts{1};
    
    try
        % 1. Load Audio (Force Mono)
        [audioData, fs] = audioread(fullFileName);
        audioData = audioData(:,1); 
        
        % Define Timing 
        winLen = round(0.03 * fs); 
        overlap = round(0.02 * fs); 
        
        % --- A. EXTRACT MFCC ---
        feats_mfcc = [];
        if useMFCC
            aFE = audioFeatureExtractor( ...
                'SampleRate', fs, ...
                'Window', hamming(winLen, 'periodic'), ... 
                'OverlapLength', overlap, ...              
                'mfcc', true);
            feats_mfcc = extract(aFE, audioData);
        end
        
        % --- B. EXTRACT PRONY FEATURES ---
        feats_prony = [];
        if useProny
            frames = buffer(audioData, winLen, overlap, 'nodelay');
            numFrames = size(frames, 2);
            
            % Local temporary variable for the loop
            tempProny = zeros(numFrames, pronyOrder * 2);
            
            for k = 1:50:numFrames
                frame = frames(:, k);
                frame = frame .* hamming(winLen, 'periodic');
                
                % Silence Check (Optimization)
                % if sum(frame.^2) < 1e-15
                %     tempProny(k, :) = 0;
                %     continue; 
                % end
                
                try
                    [b, a] = prony(frame, 0, pronyOrder);
                    poles = roots(a);
                    
                    pFreqs = abs(angle(poles)) * (fs / (2*pi));
                    pDamp = -log(abs(poles));
                    
                    % Sort poles by frequency
                    [pFreqs, sortIdx] = sort(pFreqs);
                    pDamp = pDamp(sortIdx);
                    
                    rowFeat = zeros(1, pronyOrder * 2);
                    rowFeat(1:2:end) = pFreqs(1:pronyOrder);
                    rowFeat(2:2:end) = pDamp(1:pronyOrder);
                    
                    tempProny(k, :) = rowFeat;
                catch
                    tempProny(k, :) = NaN;
                end
            end
            feats_prony = tempProny;
        end
        
        % --- C. SYNCHRONIZE & MERGE ---
        nRows = min(size(feats_mfcc, 1), size(feats_prony, 1));
        if isempty(feats_mfcc), nRows = size(feats_prony, 1); end
        if isempty(feats_prony), nRows = size(feats_mfcc, 1); end
        
        fileFeatures = [];
        if useMFCC, fileFeatures = [fileFeatures, feats_mfcc(1:nRows, :)]; end
        if useProny, fileFeatures = [fileFeatures, feats_prony(1:nRows, :)]; end
        
        % --- STORE IN CELL ARRAYS (THE FIX) ---
        % Writing to two different sliced variables is allowed.
        featResults{i} = fileFeatures;
        labelResults{i} = repmat({trueLabel}, nRows, 1);
        
    catch ME
        fprintf('Error processing file %s: %s\n', baseFileName, ME.message);
    end
end 

% --- POST-PROCESSING ---
disp('Combining parallel results...');

% Remove empty entries (failed files)
emptyIdx = cellfun(@isempty, featResults);
featResults(emptyIdx) = [];
labelResults(emptyIdx) = [];

% Vertically concatenate all cell contents
allFeatures = vertcat(featResults{:});
allLabels = vertcat(labelResults{:});

disp('Removing silent frames...');
totalFrames = size(allFeatures, 1);

% Count specific issues
isNaNRow = any(isnan(allFeatures), 2);
isZeroRow = sum(abs(allFeatures), 2) < 1e-12; % Effectively zero

fprintf('Total Frames Extracted: %d\n', totalFrames);
fprintf('Frames failed (Prony Error): %d (%.2f%%)\n', sum(isNaNRow), (sum(isNaNRow)/totalFrames)*100);
fprintf('Frames skipped (Silence):    %d (%.2f%%)\n', sum(isZeroRow), (sum(isZeroRow)/totalFrames)*100);

% Remove Bad Data
toRemove = isNaNRow | isZeroRow;
allFeatures(toRemove, :) = [];
allLabels(toRemove) = [];

fprintf('Final Valid Frames: %d\n', size(allFeatures, 1));

disp('Sanitizing and Normalizing...');
allFeatures(isinf(allFeatures)) = 0;
allFeatures(isnan(allFeatures)) = 0;
featuresNorm = normalize(allFeatures);

% --- DOWNSAMPLING ---
disp('Downsampling for visualization speed...');
downsampleIdx = 1:10:size(featuresNorm, 1);
featuresSub = featuresNorm(downsampleIdx, :);
labelsSub = allLabels(downsampleIdx);

% --- T-SNE ---
disp('Running t-SNE...');
Y = tsne(featuresSub, 'NumDimensions', 2);

% --- PLOTTING ---
disp('Plotting...');
figure;
gscatter(Y(:,1), Y(:,2), labelsSub);

featureString = strjoin(activeFeatureNames, ' + ');
title(['t-SNE Visualization using: ' featureString]);
xlabel('t-SNE Dim 1');
ylabel('t-SNE Dim 2');
grid on;
legend('Location', 'bestoutside');