clear; clc; close all;

% --- 1. CONFIGURATION ---
parquetDataPath = 'drone-audio-detection-samples/data';
holdoutRatio = 0.2; 

% Row Splitting
targetRowsPerTask = 1000; 

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
customConfig.stdProny = true;        
customConfig.dampProny = true;      
customConfig.freqProny = true;
customConfig.settling = true;

% Settling Mode Control (Choose 1 of 3 options)
% 'drop'          - Original behavior: drops the first 0.3s, processes the rest.
% 'bidirectional' - Keeps EVERYTHING: looks forward for first 0.3s, backward for the rest.
% 'early_only'    - Keeps ONLY the first 0.3s (looks forward), drops the rest.
customConfig.settlingMode = 'early_only';

% --- 2. FILE DISCOVERY & ROW CHUNKING ---
fprintf('Scanning files for row-based splitting...\n');
rawFiles = dir(fullfile(parquetDataPath, '*.parquet'));
if isempty(rawFiles)
    error('No .parquet files found in %s', parquetDataPath);
end

taskList = struct('File', {}, 'Folder', {}, 'StartRow', {}, 'EndRow', {}, 'TaskID', {});

for i = 1:length(rawFiles)
    fullPath = fullfile(rawFiles(i).folder, rawFiles(i).name);
    numRowsTotal = 0;
    try
        pInfo = parquetinfo(fullPath);
        numRowsTotal = pInfo.NumRows;
    catch
        try
            T_preview = parquetread(fullPath); 
            numRowsTotal = height(T_preview);
        catch ME
            fprintf('Warning: Could not read %s. Reason: %s\n', rawFiles(i).name, ME.message);
            continue; 
        end
    end
    
    if numRowsTotal > 0
        starts = 1:targetRowsPerTask:numRowsTotal;
        ends = [starts(2:end)-1, numRowsTotal];
        for k = 1:length(starts)
            taskList(end+1) = struct('File', rawFiles(i).name, ...
                                     'Folder', rawFiles(i).folder, ...
                                     'StartRow', starts(k), ...
                                     'EndRow', ends(k), ...
                                     'TaskID', sprintf('%s_Part%d', rawFiles(i).name, k));
        end
    end
end

numTotalTasks = length(taskList);
fprintf('Found %d Files. Created %d Tasks.\n', length(rawFiles), numTotalTasks);

% --- 3. FEATURE NAMES SETUP ---
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
% Note: Custom Names added here for consistency, though order depends on loop below
if customConfig.bicoherence
    varNames = [varNames, {'AIB_Heli_0_100Hz', 'AIB_Drone_100_300Hz', ...
        'AIB_Harmonic_300_1k', 'AIB_Mid_1k_5k', 'AIB_Empty_5k_10k', 'AIB_PWM_10k_Plus'}];
end
if customConfig.tkeo
    varNames = [varNames, {'TKEO_Mean', 'TKEO_Std', 'TKEO_Max', 'TKEO_Kurtosis'}];
end
% Fixed Order for Prony: STD, Freq, Damp
if customConfig.stdProny
    for ii = 1:8, varNames{end+1} = sprintf('STD_Prony_Freq_%d', ii); end      
end 
if customConfig.freqProny
    for ii = 1:8, varNames{end+1} = sprintf('Prony_Freq_%d', ii); end
end 
if customConfig.dampProny
    for ii = 1:8, varNames{end+1} = sprintf('Prony_Damp_%d', ii); end 
end

% --- 4. PARALLEL PROCESSING ---
currentPool = gcp('nocreate');
targetWorkers = 20; 

if isempty(currentPool)
    parpool('local', targetWorkers);
elseif currentPool.NumWorkers < targetWorkers
    delete(currentPool);
    parpool('local', targetWorkers);
end

dq = parallel.pool.DataQueue;
afterEach(dq, @(msg) fprintf('%s', msg));

fprintf('\n--- STARTING ROW-BASED PARALLEL PROCESSING ---\n');
resultsBuffer = cell(numTotalTasks, 1);

parfor taskIdx = 1:numTotalTasks
    taskTic = tic; 
    taskDef = taskList(taskIdx);
    
    try
        fullPath = fullfile(taskDef.Folder, taskDef.File);
        
        % --- LOAD PARQUET ---
        T = [];
        try
            rf = rowfilter("RowIndex");
            rf = (rf >= taskDef.StartRow) & (rf <= taskDef.EndRow);
            T = parquetread(fullPath, "RowFilter", rf);
        catch
             T = parquetread(fullPath);
             T = T(taskDef.StartRow:min(height(T), taskDef.EndRow), :);
        end
        
        numRows = height(T);
        
        % --- EXTRACT RAW AUDIO ---
        audioBatch = cell(numRows, 1);
        filenameBatch = cell(numRows, 1);
        labelBatch = cell(numRows, 1);
        fs = 16000; 
        
        if ismember('audio', T.Properties.VariableNames), rawCol = T.audio; else, rawCol = T; end
        hasLabel = ismember('label', T.Properties.VariableNames);

        for r = 1:numRows
            % Unwrap complex parquet cell structures
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
            
            absRow = taskDef.StartRow + r - 1;
            filenameBatch{r} = sprintf('%s_Row%d', taskDef.File, absRow);
            
            if hasLabel
                rawL = T.label(r);
                if iscell(rawL), rawL=rawL{1}; end
                if isnumeric(rawL)
                    if rawL==1, labelBatch{r}='DRONE'; else, labelBatch{r}='OTHER'; end
                else, labelBatch{r} = char(rawL); end
            else, labelBatch{r} = 'OTHER'; end
        end
        
        % --- PROCESS FEATURES (SYNCHRONIZED LOOP) ---
        taskFeats = []; taskLabels = {}; taskNames = {};
        
        for j = 1:length(audioBatch)
            audioData = audioBatch{j};
            currentLabel = labelBatch{j};
            currentName = filenameBatch{j};
            
            % Window Settings (30ms Window, 10ms Hop)
            winLen = round(0.03 * fs);        
            overlap = round(0.02 * fs);       
            hop = winLen - overlap;
            
            % Standard Deviation Settings (0.3s Settling Time)
            std_window_sec = 0.3;
            min_history_frames = round(std_window_sec / (hop/fs)); % Approx 30 frames
            
            % 1. Calculate Standard AFE for Whole File first
            % We do this to get the exact frame alignment logic of MATLAB
            file_afe_feats = [];
            if hasStandardFeatures
                aFE = audioFeatureExtractor('SampleRate',fs, 'Window',hamming(winLen,'periodic'), 'OverlapLength',overlap);
                set(aFE, afeConfig);
                file_afe_feats = extract(aFE, audioData);
            else
                % Dummy frame count calculation if no AFE
                num_frames_calc = floor((length(audioData) - winLen) / hop) + 1;
                file_afe_feats = zeros(num_frames_calc, 0);
            end
            
            numFrames = size(file_afe_feats, 1);
            
            % 2. Pre-allocate Arrays for Custom Instantaneous Features
            all_basics = cell(numFrames, 1);
            all_freqs  = zeros(numFrames, 8);
            all_damps  = zeros(numFrames, 8);
            valid_frames_count = 0;
            
            % Optimize Pass 1: if we only want early frames, don't compute the whole file
            max_pass1_frames = numFrames;
            if strcmp(customConfig.settlingMode, 'early_only')
                max_pass1_frames = min(numFrames, 2 * min_history_frames - 1);
            end
            
            % --- 3A. First Pass: Extract Instantaneous Features ---
            for k = 1:max_pass1_frames
                sIdx = (k-1)*hop + 1;
                eIdx = sIdx + winLen - 1;
                
                if eIdx > length(audioData), break; end
                chunk = audioData(sIdx:eIdx);
                
                row_custom_inst = [];
                if customConfig.bicoherence
                    row_custom_inst = [row_custom_inst, getBicoherenceFeature(chunk, fs)];
                end
                if customConfig.tkeo
                    row_custom_inst = [row_custom_inst, getNormTKEOFeatures(chunk)];
                end
                all_basics{k} = row_custom_inst;
                
                if customConfig.stdProny || customConfig.dampProny || customConfig.freqProny
                    [freq_map, amp_map, damp_map, time_vec, win_len, num_of_win] = prony_tracker(chunk, fs, 0.03);
                    [~, inst_damp, inst_freq] = get_features_from_prony(freq_map, amp_map, damp_map, time_vec, win_len, num_of_win);
                    
                    all_freqs(k, :) = inst_freq.';
                    all_damps(k, :) = inst_damp.';
                end
                valid_frames_count = k;
            end
            
            % Trim features to match processed chunks
            numFrames = valid_frames_count;
            file_afe_feats = file_afe_feats(1:numFrames, :);
            valid_feats_for_file = [];
            
            % Optimize Pass 2: restrict output to just the early frames if requested
            max_pass2_frames = numFrames;
            if strcmp(customConfig.settlingMode, 'early_only')
                max_pass2_frames = min(numFrames, min_history_frames - 1);
            end
            
            % --- 3B. Second Pass: Rolling Features based on Settling Mode ---
            for k = 1:max_pass2_frames
                use_frame = true;
                row_custom = all_basics{k};
                
                if customConfig.stdProny || customConfig.dampProny || customConfig.freqProny
                    
                    % Determine the Rolling Window Bounds
                    if k >= min_history_frames
                        % Normal Forward Settling: Uses the PAST 0.3 seconds
                        hist_idx = (k - min_history_frames + 1) : k;
                    else
                        % Early frames (< 0.3s)
                        if strcmp(customConfig.settlingMode, 'drop')
                            use_frame = false;
                            hist_idx = []; 
                        else 
                            % 'bidirectional' or 'early_only'
                            % Reverse Settling: Uses the FUTURE 0.3 seconds
                            hist_idx = k : min(numFrames, k + min_history_frames - 1);
                        end
                    end
                    
                    if use_frame
                        % Calculate Standard Deviation over the dynamic window
                        feat_std = std(all_freqs(hist_idx, :), 0, 1);
                        
                        prony_vec = [];
                        if customConfig.stdProny,  prony_vec = [prony_vec, feat_std]; end
                        if customConfig.freqProny, prony_vec = [prony_vec, all_freqs(k, :)]; end
                        if customConfig.dampProny, prony_vec = [prony_vec, all_damps(k, :)]; end
                        
                        row_custom = [row_custom, prony_vec];
                    end
                end
                
                % Aggregate Valid Frame
                if use_frame
                    row_afe = file_afe_feats(k, :);
                    valid_feats_for_file = [valid_feats_for_file; [row_afe, row_custom]];
                end
            end
            
            % Store Result for this Audio File
            if ~isempty(valid_feats_for_file)
                rows_to_add = size(valid_feats_for_file, 1);
                taskFeats = [taskFeats; valid_feats_for_file];
                taskLabels = [taskLabels; repmat({currentLabel}, rows_to_add, 1)];
                taskNames = [taskNames; repmat({currentName}, rows_to_add, 1)];
            end
        end
        
        S = struct('X', taskFeats, 'Y', {taskLabels}, 'N', {taskNames});
        resultsBuffer{taskIdx} = S;
        
        duration = toc(taskTic);
        send(dq, sprintf('[DONE] Task %d/%d (%s) rows %d-%d in %.2fs.\n', ...
            taskIdx, numTotalTasks, taskDef.File, taskDef.StartRow, taskDef.EndRow, duration));
        
    catch ME
        send(dq, sprintf('[ERROR] Task %d (%s) failed: %s\n', taskIdx, taskDef.TaskID, ME.message));
    end
end

% --- 5. AGGREGATE ---
fprintf('Aggregating results from %d tasks...\n', numTotalTasks);
X_All = []; Y_All = []; N_All = [];
for i = 1:numTotalTasks
    if ~isempty(resultsBuffer{i})
        X_All = [X_All; resultsBuffer{i}.X];
        Y_All = [Y_All; resultsBuffer{i}.Y];
        N_All = [N_All; resultsBuffer{i}.N];
    end
end

if isempty(X_All)
    error('No features extracted. Check paths or error logs.');
end

% Check feature count matches name count
if size(X_All, 2) ~= length(varNames)
    warning('Dimension mismatch: Data has %d cols, Names has %d. Trimming/Padding Names.', size(X_All,2), length(varNames));
    if length(varNames) > size(X_All, 2)
        varNames = varNames(1:size(X_All, 2));
    else
        % Pad
        for x = (length(varNames)+1):size(X_All,2)
            varNames{end+1} = sprintf('ExtraF_%d', x);
        end
    end
end

FullTable = array2table(X_All, 'VariableNames', varNames);
FullTable.Label = categorical(string(Y_All));
FullTable.Filename = string(N_All);

% --- 6. SPLIT & SAVE (FIXED) ---
fprintf('Splitting Train/Test...\n');
origFiles = regexprep(FullTable.Filename, '_Row\d+$', '');
uniqueFiles = unique(origFiles);
numFiles = length(uniqueFiles);

% --- SAVE TO DISK ---
fprintf('Saving FullTable to disk...\n');
save('RevTrueFullTable.mat', 'FullTable', '-v7.3');
fprintf('Saved FullTable.mat successfully.\n');
