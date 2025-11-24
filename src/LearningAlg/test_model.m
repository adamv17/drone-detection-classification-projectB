% --- CONFIGURATION ---
maxPlots = 5; 
plotCount = 0;
K_threshold = 2.0; % Factor K for the median filter (Adjust if needed)

% We rely on featResults from your workspace to avoid re-extraction
if ~exist('featResults', 'var') || ~exist('wavFiles', 'var')
    error('Missing "featResults" or "wavFiles". Please run the feature extraction code first.');
end

disp('------------------------------------------------');
disp('VISUALIZING: FILTERED STFT vs CLASSIFICATION');
disp('(Using workspace features | Adaptive Median Filter)');
disp('------------------------------------------------');

% --- MODEL HANDLER: Detect if it's a Struct or Object ---
modelIsStruct = isstruct(trainedCosineKNNModel);

for i = 1:length(wavFiles)
    
    % --- FILTER: TEST SET ---
    if ~isTestFile(i) 
        continue; 
    end
    
    fileEntry = wavFiles(i);
    nameParts = split(fileEntry.name, '_');
    
    % --- FILTER: DRONES ONLY (Case Insensitive) ---
    if ~strcmpi(nameParts{1}, 'Drone')
        continue;
    end 
    
    if plotCount >= maxPlots
        fprintf('Stopping after %d plots.\n', maxPlots);
        break;
    end
    plotCount = plotCount + 1;

    % 1. GET FEATURES FROM WORKSPACE
    rawFeatures = featResults{i};
    if isempty(rawFeatures), continue; end
    
    % 2. PREPARE TABLE FOR MODEL
    numFeats = size(rawFeatures, 2);
    allNames = arrayfun(@(x) sprintf('MFCC_%d', x), 1:numFeats, 'UniformOutput', false);
    tempTable = array2table(rawFeatures, 'VariableNames', allNames);
    
    % 3. PREDICT (ROBUST METHOD)
    try
        if modelIsStruct
            if isfield(trainedCosineKNNModel, 'predictFcn')
                preds = trainedCosineKNNModel.predictFcn(tempTable);
            else
                 try
                    tempTable(:, {'MFCC_1', 'MFCC_6', 'MFCC_12'}) = [];
                 catch
                 end
                 fields = fieldnames(trainedCosineKNNModel);
                 internalModel = trainedCosineKNNModel.(fields{1}); 
                 preds = predict(internalModel, tempTable);
            end
        else
            try
                tempTable(:, {'MFCC_1', 'MFCC_6', 'MFCC_12'}) = [];
            catch
            end
            preds = predict(trainedCosineKNNModel, tempTable);
        end
    catch ME
        warning('Prediction failed for %s: %s', fileEntry.name, ME.message);
        continue;
    end
    
    % Convert to Binary (1=Drone, 0=Other)
    isDrone = (preds == "DRONE") | strcmp(string(preds), "DRONE");
    binaryPred = double(isDrone);

    % 4. LOAD AUDIO
    [audioData, fs] = audioread(fullfile(dataPath, fileEntry.name));
    if size(audioData, 2) > 1, audioData = mean(audioData, 2); end

    % 5. CALCULATE TIME AXES
    % A. For Spectrogram (Will be calculated by spectrogram function later, but we need max time)
    t_max = (length(audioData)-1)/fs;
    
    % B. For Prediction (Aligned to MFCC windows)
    winLen = round(0.03 * fs); 
    hopLength = winLen - round(0.02 * fs); 
    numWindows = size(rawFeatures, 1);
    t_pred = ((0:numWindows-1) * hopLength + (winLen/2)) / fs;
    
    % 6. VISUALIZE
    figure('Name', ['File: ', fileEntry.name], 'Color', 'w', 'Position', [100, 100, 1000, 700]);
    
    % --- SUBPLOT 1: FILTERED SPECTROGRAM (Custom Logic) ---
    ax1 = subplot(2,1,1);
    
    % custom stft parameters
    win_viz = 4096;
    ov_viz = win_viz / 2;
    nfft_viz = 4096;
    
    % 1. Compute STFT
    [S, F, T] = spectrogram(audioData, win_viz, ov_viz, nfft_viz, fs);
    S_mag = abs(S);
    
    % 2. Adaptive Median Threshold
    % Filter along frequency axis (dim 1)
    baseline = medfilt1(S_mag, 5, [], 1); 
    threshold = baseline *1.2;
    
    % 3. Apply Mask
    mask = S_mag >= threshold;
    S_filtered = S_mag .* mask;
    
    % 4. Plot (using imagesc)
    % We add eps to avoid log(0) = -Inf
    imagesc(T, F, 20*log10(S_filtered + eps)); 
    axis xy; % Correct orientation
    colormap('jet');
    colorbar;
    
    title(['Filtered STFT (K=', num2str(K_threshold), '): ', fileEntry.name], 'Interpreter', 'none');
    ylabel('Frequency (Hz)');
    
    % --- SUBPLOT 2: CLASSIFICATION RESULT ---
    ax2 = subplot(2,1,2);
    stairs(t_pred, binaryPred, 'LineWidth', 2, 'Color', '#D95319'); 
    
    grid on;
    ylim([-0.2, 1.2]); 
    yticks([0 1]);
    yticklabels({'OTHER', 'DRONE'});
    xlabel('Time (s)');
    title('Model Classification');
    
    % Link axes for zooming
    linkaxes([ax1, ax2], 'x');
    xlim([0, t_max]);
end