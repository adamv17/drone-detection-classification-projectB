clear; clc; close all;

% --- CONFIGURATION ---
dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 
useMFCC = true;
useProny = false;
pronyOrder = 20; 

% Initialize lists for dynamic title
activeFeatureNames = {};
if useMFCC, activeFeatureNames{end+1} = 'MFCC'; end
if useProny, activeFeatureNames{end+1} = sprintf('Prony(Order=%d)', pronyOrder); end

wavFiles = dir(fullfile(dataPath, '*.wav'));
numFiles = length(wavFiles);

% --- PRE-ALLOCATE STORAGE ---
featResults = cell(numFiles, 1);
labelResults = cell(numFiles, 1);

disp(['Starting parallel extraction on ' num2str(numFiles) ' files...']);

parfor i = 1:numFiles
    % Re-construct file info inside the loop
    fileEntry = wavFiles(i);
    baseFileName = fileEntry.name;
    fullFileName = fullfile(dataPath, baseFileName);
    
    % Extract Label (Assuming format Label_...)
    nameParts = split(baseFileName, '_');
    trueLabel = nameParts{1};
    
    try
        % 1. Load Audio (Force Mono)
        [audioData, fs] = audioread(fullFileName);
        if size(audioData, 2) > 1
            audioData = mean(audioData, 2); % Convert to mono if stereo
        end
        
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
        
        % --- B. EXTRACT PRONY FEATURES (WITH AMPLITUDES) ---
        feats_prony = [];
        if useProny
            % Create frames
            frames = buffer(audioData, winLen, overlap, 'nodelay');
            numFrames = size(frames, 2);
            
            % Size: Freq + Damp + Amp = 3 features per pole
            numFeaturesPerPole = 3; 
            tempProny = zeros(numFrames, pronyOrder * numFeaturesPerPole);
            
            % Create time vector for Least Squares (0 to N-1)
            tVec = (0:winLen-1)'; 
            
            % ITERATE ALL FRAMES (Removed the 1:50:numFrames skip)
            for k = 1:numFrames
                frame = frames(:, k);
                win = hamming(winLen, 'periodic');
                frameWindowed = frame .* win;
                
                try
                    % 1. Find Poles (Frequency & Damping)
                    [b, a] = prony(frameWindowed, 0, pronyOrder);
                    poles = roots(a);
                    
                    % 2. THE MISSING STEP: Find Amplitudes (Residues)
                    % We solve: V * Amps = Frame
                    % V is a matrix where col j is (pole_j)^t
                    % We use the raw frame for amplitude fitting, not windowed, 
                    % usually better for physical parameter estimation, 
                    % but windowed is safer for stability. Let's stick to windowed.
                    
                    % Construct Vandermonde Matrix manually or via broadcasting
                    % V dimensions: [winLen x pronyOrder]
                    V = (poles.').^tVec; 
                    
                    % Least Squares Solve for Complex Amplitudes
                    complexAmps = V \ frameWindowed;
                    
                    % 3. Convert to Physical Parameters
                    pFreqs = abs(angle(poles)) * (fs / (2*pi));
                    pDamp = -log(abs(poles)); % Positive value = stable decay
                    pAmps = abs(complexAmps); % Magnitude of the residue
                    
                    % 4. Sort by Frequency (to keep features aligned)
                    [pFreqs, sortIdx] = sort(pFreqs);
                    pDamp = pDamp(sortIdx);
                    pAmps = pAmps(sortIdx);
                    
                    % 5. Pack into Feature Row
                    rowFeat = zeros(1, pronyOrder * numFeaturesPerPole);
                    
                    % Pattern: [F1, D1, A1, F2, D2, A2, ...]
                    rowFeat(1:3:end) = pFreqs(1:pronyOrder);
                    rowFeat(2:3:end) = pDamp(1:pronyOrder);
                    rowFeat(3:3:end) = pAmps(1:pronyOrder);
                    
                    tempProny(k, :) = rowFeat;
                catch
                    % If Prony fails (unstable or singular), mark NaN
                    tempProny(k, :) = NaN;
                end
            end
            feats_prony = tempProny;
        end
        
        % --- C. SYNCHRONIZE & MERGE ---
        if useMFCC && useProny
            nRows = min(size(feats_mfcc, 1), size(feats_prony, 1));
        elseif useMFCC
            nRows = size(feats_mfcc, 1);
        elseif useProny
            nRows = size(feats_prony, 1);
        else
            nRows = 0;
        end
        
        fileFeatures = [];
        if nRows > 0
            if useMFCC, fileFeatures = [fileFeatures, feats_mfcc(1:nRows, :)]; end
            if useProny, fileFeatures = [fileFeatures, feats_prony(1:nRows, :)]; end
            
            featResults{i} = fileFeatures;
            labelResults{i} = repmat({trueLabel}, nRows, 1);
        end
        
    catch ME
        fprintf('Error processing file %s: %s\n', baseFileName, ME.message);
    end
end 

% --- POST-PROCESSING ---
disp('Combining parallel results...');
emptyIdx = cellfun(@isempty, featResults);
featResults(emptyIdx) = [];
labelResults(emptyIdx) = [];

allFeatures = vertcat(featResults{:});
allLabels = vertcat(labelResults{:});

disp('Cleaning data...');
totalFrames = size(allFeatures, 1);

% Remove NaN rows (Prony failures)
allFeatures(isnan(allFeatures)) = 0;
allFeatures(isinf(allFeatures)) = 0;

% Remove Silence (Exact zeros from initialization)
% Since we have amplitudes now, we can check if sum(Amps) is tiny
if ~isempty(allFeatures)
    isZeroRow = sum(abs(allFeatures), 2) < 1e-6;
    allFeatures(isZeroRow, :) = [];
    allLabels(isZeroRow) = [];
end

fprintf('Final Valid Frames: %d (from %d)\n', size(allFeatures, 1), totalFrames);

disp('Normalizing...');
featuresNorm = normalize(allFeatures);

% --- DOWNSAMPLING FOR PLOT ---
disp('Downsampling for visualization...');
if size(featuresNorm, 1) > 5000
    downsampleIdx = 1:20:size(featuresNorm, 1);
else
    downsampleIdx = 1:size(featuresNorm, 1);
end
featuresSub = featuresNorm(downsampleIdx, :);
labelsSub = allLabels(downsampleIdx);

% --- T-SNE ---
disp('Running t-SNE...');
% Try-catch for t-SNE in case of perplexity issues with small data
try
    Y = tsne(featuresSub, 'NumDimensions', 2);
    
    disp('Plotting...');
    figure;
    gscatter(Y(:,1), Y(:,2), labelsSub);
    featureString = strjoin(activeFeatureNames, ' + ');
    title(['t-SNE: ' featureString ' (Freq/Damp/Amp)']);
    xlabel('t-SNE Dim 1');
    ylabel('t-SNE Dim 2');
    grid on;
    legend('Location', 'bestoutside');
catch ME
    disp('t-SNE failed (probably not enough distinct data points):');
    disp(ME.message);
end