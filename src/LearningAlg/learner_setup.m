%% learner_setup.m
clear; clc; close all;

% --- 1. CONFIGURATION ---
dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 
holdoutRatio = 0.2; 

% AFE Settings
afeConfig.mfcc = true;  
afeConfig.mfccDelta = false;
afeConfig.spectralCentroid = false;  
afeConfig.spectralRolloffPoint = false; 
afeConfig.spectralFlux = false;      
afeConfig.zerocrossrate = false;  
afeConfig.pitch = false;
afeConfig.harmonicRatio = true;
afeConfig.spectralKurtosis = true;

% Custom Settings
customConfig.bicoherence = false; 
customConfig.tkeo = true;
customConfig.stdProny = false;
customConfig.dampProny = true;
customConfig.freqProny = true;

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
if customConfig.stdProny
    for ii = 1:8
        varNames{end+1} = sprintf('STD_Prony_Freq_%d', ii);
    end    
end 
if customConfig.freqProny
    for ii = 1:8
        varNames{end+1} = sprintf('Prony_Freq_%d', ii);
    end
end 

if customConfig.dampProny
    for ii = 1:8
        varNames{end+1} = sprintf('Prony_Damp_%d', ii);
    end 
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
            longWinSec = 0.5; % sec
            longWin = round(longWinSec * fs);
            
            for k = 1:numFrames
                % Map frame index to sample center
                currentCenter = round((k-1)*hop + (longWin/2));
                
                % Extract short window (standard 30ms is fine for TKEO)
                sIdx = currentCenter - floor(longWin/2);
                eIdx = sIdx + longWin - 1;
                
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
        
        % --- PRONY SECTION: THE "3 SHIFTS" APPROACH ---
        if customConfig.stdProny || customConfig.dampProny || customConfig.freqProny
            
            % 1. Setup Parameters
            prony_win_sec = 0.03;      % 30 ms physics window
            shift_samples = hop;       % 10 ms shift (to match MFCC)
            prony_order = 50; 
            
            % We will accumulate results from 3 passes here
            % Columns: [Time, Freq1..8, Damp1..8]
            all_tracks = [];
            
            % 2. Run Prony 3 times (0ms, 10ms, 20ms offsets)
            % This aligns the 30ms-stepped tracker with our 10ms grid
            for offset_i = 0:2
                
                start_samp = 1 + (offset_i * shift_samples);
                if start_samp > length(audioData), break; end
                
                audio_shifted = audioData(start_samp:end);
                
                % SAFETY: Ensure audio is long enough for at least one window
                if length(audio_shifted) < round(prony_win_sec * fs)
                    continue; 
                end

                try
                    % Run Tracker on the LONG audio
                    [f_map, amp_map, d_map, t_vec, w_len, n_win] = prony_tracker(audio_shifted, fs, prony_win_sec);
                    
                    % Extract Matrices
                    [~, d_mat, f_mat] = get_features_from_prony(f_map, amp_map, d_map, t_vec, w_len, n_win);
                    
                    % Create Time Vector adjusted for the offset
                    % t_vec from tracker starts at 0 relative to audio_shifted
                    real_time = t_vec + (start_samp - 1)/fs;

                    d_mat = d_mat.';
                    f_mat = f_mat.';
                    
                    % Ensure Dimensions (N x 8)
                    if size(f_mat, 2) > 8, f_mat = f_mat(:, 1:8); end
                    if size(d_mat, 2) > 8, d_mat = d_mat(:, 1:8); end
                    if size(f_mat, 2) < 8, f_mat = [f_mat, zeros(size(f_mat,1), 8-size(f_mat,2))]; end
                    if size(d_mat, 2) < 8, d_mat = [d_mat, zeros(size(d_mat,1), 8-size(d_mat,2))]; end
                    
                    % Append to collection
                    % [Time, Freqs(8), Damps(8)]
                    batch_res = [real_time(:), f_mat, d_mat];
                    all_tracks = [all_tracks; batch_res];
                    
                catch ME
                    warning(ME.message)
                end
            end
            
            % 3. Sort and Interpolate (ROBUST FIX)
            if customConfig.stdProny
                if isempty(all_tracks)
                    final_prony_feats = zeros(numFrames, 24);
                else
                    % A. Remove NaNs in Time
                    mask = ~isnan(all_tracks(:,1));
                    all_tracks = all_tracks(mask, :);
                    
                    % B. Round Time to 5 decimal places (Fixes Jitter)
                    all_tracks(:,1) = round(all_tracks(:,1), 5);
                    
                    % C. Sort
                    [~, sortIdx] = sort(all_tracks(:,1));
                    sorted_tracks = all_tracks(sortIdx, :);
                    
                    % D. Unique (Removes Duplicates after rounding)
                    [uTimes, uIdx] = unique(sorted_tracks(:, 1));
                    sorted_tracks = sorted_tracks(uIdx, :);
                    
                    % E. Interpolate
                    if length(uTimes) < 2
                        % Not enough points to interpolate
                        final_prony_feats = zeros(numFrames, 24);
                    else
                        target_times = ((0:numFrames-1) * hop + (winLen/2)) / fs;
                        
                        interp_freq = interp1(uTimes, sorted_tracks(:,2:9), target_times, 'nearest', 'extrap');
                        interp_damp = interp1(uTimes, sorted_tracks(:,10:17), target_times, 'nearest', 'extrap');
                        
                        interp_freq(isnan(interp_freq)) = 0;
                        interp_damp(isnan(interp_damp)) = 0;
                        
                        std_window_sec = 0.5;
                        std_window_frames = max(1, round(std_window_sec / (hop/fs))); 
                        interp_std = movstd(interp_freq, [std_window_frames 0], 1); 
    
                        final_prony_feats = [interp_freq, interp_damp, interp_std];
                    end
                end
            else
                final_prony_feats = all_tracks(:,2:17)
            end 
          
            
            % Stitch
            if size(final_prony_feats, 1) > numFrames
                final_prony_feats = final_prony_feats(1:numFrames, :);
            elseif size(final_prony_feats, 1) < numFrames
                pad = repmat(final_prony_feats(end,:), numFrames - size(final_prony_feats,1), 1);
                final_prony_feats = [final_prony_feats; pad];
            end
            
            custom_features = [custom_features, final_prony_feats];
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
clearvars -except TrainTable TestTable FullTable afeConfig customConfig;