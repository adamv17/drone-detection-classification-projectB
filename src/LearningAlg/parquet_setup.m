%% learner_setup_parfor_test.m
clear; clc; close all;

% --- 1. CONFIGURATION ---
parquetDataPath = '../../datasets/drone-audio-detection-samples/data';
holdoutRatio = 0.2; 

% AFE Settings
afeConfig.mfcc = true;  
afeConfig.mfccDelta = false;
afeConfig.harmonicRatio = true;
afeConfig.spectralKurtosis = true;
afeConfig.spectralCentroid = false;  
afeConfig.spectralRolloffPoint = false; 
afeConfig.spectralFlux = false;      
afeConfig.zerocrossrate = false;  
afeConfig.pitch = false;

% Custom Features
customConfig.bicoherence = false;   
customConfig.tkeo = false;          
customConfig.stdProny = false;      
customConfig.dampProny = true;     
customConfig.freqProny = true;

% --- 2. FILE DISCOVERY ---
subTasks = dir(fullfile(parquetDataPath, '*.parquet'));
if isempty(subTasks)
    error('No .parquet files found.');
end
numTasks = length(subTasks);
fprintf('Found %d Parquet files.\n', numTasks);

% --- 3. FEATURE NAMES ---
varNames = {};
afeValues = struct2cell(afeConfig);
hasStandardFeatures = any([afeValues{:}]); 

if hasStandardFeatures
    dummyFs = 16000; 
    dummyAFE = audioFeatureExtractor('SampleRate', dummyFs, ...
        'Window', hamming(round(0.03*dummyFs)), 'OverlapLength', round(0.02*dummyFs));
    set(dummyAFE, afeConfig);
    infoSt = info(dummyAFE);
    fNames = fieldnames(infoSt);
    for i = 1:numel(fNames)
        fn = fNames{i};
        if numel(infoSt.(fn)) == 1, varNames{end+1} = fn;
        else, for k=1:numel(infoSt.(fn)), varNames{end+1} = sprintf('%s_%d',fn,k); end; end
    end
end
if customConfig.bicoherence
    varNames = [varNames, {'AIB_Heli_0_100Hz', 'AIB_Drone_100_300Hz', ...
        'AIB_Harmonic_300_1k', 'AIB_Mid_1k_5k', 'AIB_Empty_5k_10k', 'AIB_PWM_10k_Plus'}];
end
if customConfig.tkeo
    varNames = [varNames, {'TKEO_Mean', 'TKEO_Std', 'TKEO_Max', 'TKEO_Kurtosis'}];
end
if customConfig.stdProny
    for ii = 1:8, varNames{end+1} = sprintf('STD_Prony_Freq_%d', ii); end    
end 
if customConfig.freqProny
    for ii = 1:8, varNames{end+1} = sprintf('Prony_Freq_%d', ii); end
end 
if customConfig.dampProny
    for ii = 1:8, varNames{end+1} = sprintf('Prony_Damp_%d', ii); end 
end
fprintf('Expected Features per frame: %d\n', length(varNames));

% --- 4. PARALLEL PROCESSING ---
currentPool = gcp('nocreate');
targetWorkers = 16; 

% Smart Pool Start
if isempty(currentPool)
    parpool('local', targetWorkers);
    fprintf('Parallel Pool started with %d workers.\n', targetWorkers);
elseif currentPool.NumWorkers < targetWorkers
    delete(currentPool);
    parpool('local', targetWorkers);
    fprintf('Parallel Pool restarted with %d workers.\n', targetWorkers);
else
    fprintf('Parallel Pool active: %d workers.\n', currentPool.NumWorkers);
end

dq = parallel.pool.DataQueue;
afterEach(dq, @(msg) fprintf('%s', msg));

fprintf('\n--- STARTING PARALLEL BENCHMARK ---\n');
resultsBuffer = cell(numTasks, 1);

loopRange = 1:39; 

fprintf('Running on index range: %d to %d\n', min(loopRange), max(loopRange));

% Start Parallel Loop
parfor taskIdx = loopRange
    taskTic = tic; 
    
    % --- FIX 1: Initialize temporaries to silence warnings ---
    taskEntry = [];      
    last_calc_feat = zeros(1, 6); 
    
    try
        taskEntry = subTasks(taskIdx);
        
        % --- LOAD PARQUET ---
        fullPath = fullfile(taskEntry.folder, taskEntry.name);
        T = parquetread(fullPath); 
        numRows = height(T);
        
        audioBatch = cell(numRows, 1);
        filenameBatch = cell(numRows, 1);
        labelBatch = cell(numRows, 1);
        fs = 16000; 
        
        hasLabel = ismember('label', T.Properties.VariableNames);
        if ismember('audio', T.Properties.VariableNames), rawCol = T.audio; else, rawCol = T; end
        
        for r = 1:numRows
            % Robust Unwrap
            if istable(rawCol)
                if ismember('array', rawCol.Properties.VariableNames), val = rawCol{r, 'array'}; 
                else, val = rawCol{r, 1}; end
            elseif iscell(rawCol), val = rawCol{r};
            else, val = rawCol(r); end
            
            while iscell(val), val = val{1}; end
            if isstruct(val)
                if isfield(val, 'array'), val = val.array;
                else, f=fieldnames(val); val=val.(f{1}); end
            end
            
            sig = double(val);
            if size(sig,2)>1, sig=sig(:,1); end
            audioBatch{r} = sig;
            filenameBatch{r} = sprintf('%s_ID%d', taskEntry.name, r);
            
            if hasLabel
                rawL = T.label(r);
                if iscell(rawL), rawL=rawL{1}; end
                if isnumeric(rawL)
                    if rawL==1, labelBatch{r}='DRONE'; else, labelBatch{r}='OTHER'; end
                else, labelBatch{r} = char(rawL); end
            else, labelBatch{r} = 'OTHER'; end
        end
        
        % --- PROCESS FEATURES ---
        taskFeats = []; taskLabels = {}; taskNames = {};
        
        for j = 1:length(audioBatch)
            audioData = audioBatch{j};
            currentLabel = labelBatch{j};
            currentName = filenameBatch{j};
            
            winLen = round(0.03 * fs);       
            overlap = round(0.02 * fs);      
            hop = winLen - overlap;
            
            % 1. Standard
            afe_features = [];
            numFrames = 0;
            if hasStandardFeatures
                aFE = audioFeatureExtractor('SampleRate',fs, 'Window',hamming(winLen,'periodic'), 'OverlapLength',overlap);
                set(aFE, afeConfig);
                afe_features = extract(aFE, audioData);
                numFrames = size(afe_features, 1);
            else
                longWin = round(0.50 * fs);
                numFrames = floor((length(audioData) - longWin) / hop);
                afe_features = zeros(numFrames, 0);
            end
            
            % 2. Custom
            custom_features = [];
            
            % -- Bicoherence --
            if customConfig.bicoherence
                bico_step = round(0.25 * fs);
                longWin = round(0.50 * fs);
                this_file_custom = zeros(numFrames, 6); 
                
                % --- FIX 2: Re-initialize specific loop var ---
                last_calc_feat = zeros(1, 6);

                for k = 1:numFrames
                    currentCenter = round((k-1)*hop + (winLen/2));
                    if k == 1 || mod(currentCenter, bico_step) < hop
                        sIdx = currentCenter - floor(longWin/2);
                        eIdx = sIdx + longWin - 1;
                        if sIdx < 1, chunk = [zeros(1-sIdx, 1); audioData(1:eIdx)];
                        elseif eIdx > length(audioData), chunk = [audioData(sIdx:end); zeros(eIdx-length(audioData), 1)];
                        else, chunk = audioData(sIdx:eIdx); end
                        last_calc_feat = getBicoherenceFeature(chunk, fs);
                    end
                    this_file_custom(k, :) = last_calc_feat;
                end
                custom_features = [custom_features, this_file_custom];
            end
            
            % -- TKEO --
            if customConfig.tkeo
                this_file_tkeo = zeros(numFrames, 4);
                longWin = round(0.5 * fs);
                for k = 1:numFrames
                    currentCenter = round((k-1)*hop + (longWin/2));
                    sIdx = currentCenter - floor(longWin/2);
                    eIdx = sIdx + longWin - 1;
                    if sIdx < 1, chunk = [zeros(1-sIdx, 1); audioData(1:eIdx)];
                    elseif eIdx > length(audioData), chunk = [audioData(sIdx:end); zeros(eIdx-length(audioData), 1)];
                    else, chunk = audioData(sIdx:eIdx); end
                    this_file_tkeo(k, :) = getNormTKEOFeatures(chunk);
                end
                custom_features = [custom_features, this_file_tkeo];
            end
            
            % -- PRONY --
            if customConfig.stdProny || customConfig.dampProny || customConfig.freqProny
                prony_win_sec = 0.03;      
                shift_samples = hop;       
                all_tracks = [];
                for offset_i = 0:2
                    start_samp = 1 + (offset_i * shift_samples);
                    if start_samp > length(audioData), break; end
                    audio_shifted = audioData(start_samp:end);
                    if length(audio_shifted) < round(prony_win_sec * fs), continue; end
                    try
                        [f_map, amp_map, d_map, t_vec, w_len, n_win] = prony_tracker(audio_shifted, fs, prony_win_sec);
                        [~, d_mat, f_mat] = get_features_from_prony(f_map, amp_map, d_map, t_vec, w_len, n_win);
                        real_time = t_vec + (start_samp - 1)/fs;
                        d_mat = d_mat.'; f_mat = f_mat.';
                        if size(f_mat, 2) > 8, f_mat = f_mat(:, 1:8); end
                        if size(d_mat, 2) > 8, d_mat = d_mat(:, 1:8); end
                        if size(f_mat, 2) < 8, f_mat = [f_mat, zeros(size(f_mat,1), 8-size(f_mat,2))]; end
                        if size(d_mat, 2) < 8, d_mat = [d_mat, zeros(size(d_mat,1), 8-size(d_mat,2))]; end
                        batch_res = [real_time(:), f_mat, d_mat];
                        all_tracks = [all_tracks; batch_res];
                    catch, end
                end
                
                if customConfig.stdProny
                    if isempty(all_tracks)
                        final_prony_feats = zeros(numFrames, 24);
                    else
                        mask = ~isnan(all_tracks(:,1));
                        all_tracks = all_tracks(mask, :);
                        [uTimes, uIdx] = unique(round(all_tracks(:, 1), 5));
                        sorted_tracks = all_tracks(uIdx, :);
                        if length(uTimes) < 2
                            final_prony_feats = zeros(numFrames, 24);
                        else
                            target_times = ((0:numFrames-1) * hop + (winLen/2)) / fs;
                            interp_freq = interp1(uTimes, sorted_tracks(:,2:9), target_times, 'nearest', 'extrap');
                            interp_damp = interp1(uTimes, sorted_tracks(:,10:17), target_times, 'nearest', 'extrap');
                            interp_freq(isnan(interp_freq)) = 0; interp_damp(isnan(interp_damp)) = 0;
                            std_window_frames = max(1, round(0.5 / (hop/fs))); 
                            interp_std = movstd(interp_freq, [std_window_frames 0], 1); 
                            final_prony_feats = [interp_freq, interp_damp, interp_std];
                        end
                    end
                else
                    if isempty(all_tracks)
                        final_prony_feats = zeros(numFrames, 16);
                    else
                         raw_prony = all_tracks(:, 2:17);
                         if size(raw_prony,1) > numFrames, final_prony_feats = raw_prony(1:numFrames,:);
                         elseif size(raw_prony,1) < numFrames, pad = zeros(numFrames-size(raw_prony,1), 16); final_prony_feats = [raw_prony; pad];
                         else, final_prony_feats = raw_prony; end
                    end
                end 
                custom_features = [custom_features, final_prony_feats];
            end 
            
            % D. Store Result
            if numFrames > 0
                final_feat = [afe_features, custom_features];
                taskFeats = [taskFeats; final_feat];
                taskLabels = [taskLabels; repmat({currentLabel}, numFrames, 1)];
                taskNames = [taskNames; repmat({currentName}, numFrames, 1)];
            end
        end
        
        S = struct('X', taskFeats, 'Y', {taskLabels}, 'N', {taskNames});
        resultsBuffer{taskIdx} = S;
        
        duration = toc(taskTic);
        send(dq, sprintf('[SUCCESS] File %d (%s) finished in %.2f seconds.\n', ...
            taskIdx, taskEntry.name, duration));
        
    catch ME
        % Fix for catch block variable use
        errName = "Unknown";
        if ~isempty(taskEntry), errName = taskEntry.name; end
        send(dq, sprintf('[ERROR] File %d (%s) failed: %s\n', taskIdx, errName, ME.message));
    end
end

% --- 5. AGGREGATE ---
fprintf('Aggregating results...\n');
X_All = []; Y_All = []; N_All = [];
for i = 1:numTasks
    if ~isempty(resultsBuffer{i})
        X_All = [X_All; resultsBuffer{i}.X];
        Y_All = [Y_All; resultsBuffer{i}.Y];
        N_All = [N_All; resultsBuffer{i}.N];
    end
end
if isempty(X_All), error('No features extracted.'); end

if size(X_All, 2) ~= length(varNames)
    warning('Dimension mismatch. Auto-generating names.');
    varNames = arrayfun(@(x) sprintf('F%d',x), 1:size(X_All,2), 'UniformOutput',false);
end

FullTable = array2table(X_All, 'VariableNames', varNames);
FullTable.Label = categorical(string(Y_All));
FullTable.Filename = string(N_All);

% --- 6. SPLIT ---
fprintf('Splitting Train/Test...\n');
uniqueFiles = unique(FullTable.Filename);
rng(42);
cvFile = cvpartition(length(uniqueFiles), 'HoldOut', holdoutRatio);
TrainTable = FullTable(ismember(FullTable.Filename, uniqueFiles(training(cvFile))), :);
TestTable  = FullTable(ismember(FullTable.Filename, uniqueFiles(test(cvFile))), :);
fprintf('DONE. Train: %d frames, Test: %d frames.\n', height(TrainTable), height(TestTable));