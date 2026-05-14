%% learner_single_sample_benchmark.m
clear; clc; close all;

% --- 1. CONFIGURATION ---
parquetDataPath = '../../datasets/drone-audio-detection-samples/data';

% AFE Settings (Matches your "Prints" script)
afeConfig.mfcc = true;  
afeConfig.mfccDelta = false;
afeConfig.harmonicRatio = true;
afeConfig.spectralKurtosis = true;
afeConfig.spectralCentroid = false;  
afeConfig.spectralRolloffPoint = false; 
afeConfig.spectralFlux = false;      
afeConfig.zerocrossrate = false;  
afeConfig.pitch = false;

% Custom Features (Matches your "Prints" script - PRONY ON)
customConfig.bicoherence = false;   
customConfig.tkeo = false;          
customConfig.stdProny = false;      
customConfig.dampProny = true;     
customConfig.freqProny = true;

% --- 2. PREPARATION ---
subTasks = dir(fullfile(parquetDataPath, '*.parquet'));
if isempty(subTasks), error('No .parquet files found.'); end

% Pick a standard 73MB file (usually index 5 based on your screenshot)
targetIdx = 5;
if length(subTasks) < targetIdx, targetIdx = 1; end
targetFile = subTasks(targetIdx);
fileSizeMB = targetFile.bytes / 1024 / 1024;

fprintf('--- BENCHMARK CONFIGURATION ---\n');
fprintf('Target File:   %s\n', targetFile.name);
fprintf('File Size:     %.2f MB\n', fileSizeMB);
fprintf('Prony Feat:    %s\n', string(customConfig.dampProny));
fprintf('-----------------------------\n');

% --- 3. FEATURE NAMES (Setup dummy AFE) ---
varNames = {};
afeValues = struct2cell(afeConfig);
if any([afeValues{:}])
    dummyFs = 16000; 
    dummyAFE = audioFeatureExtractor('SampleRate', dummyFs, ...
        'Window', hamming(round(0.03*dummyFs)), 'OverlapLength', round(0.02*dummyFs));
    set(dummyAFE, afeConfig);
end

% --- 4. LOAD ONE ROW ---
fprintf('Loading file header... ');
fullPath = fullfile(targetFile.folder, targetFile.name);
T = parquetread(fullPath); 
numRowsInFile = height(T);
fprintf('Done. (File contains %d audio rows)\n', numRowsInFile);

% Extract Row 1 Audio
if ismember('audio', T.Properties.VariableNames), rawCol = T.audio; else, rawCol = T; end
rowIdx = 1;

% Robust Unwrap for Row 1
if istable(rawCol)
    if ismember('array', rawCol.Properties.VariableNames), val = rawCol{rowIdx, 'array'}; 
    else, val = rawCol{rowIdx, 1}; end
elseif iscell(rawCol), val = rawCol{rowIdx};
else, val = rawCol(rowIdx); end

while iscell(val), val = val{1}; end
if isstruct(val)
    if isfield(val, 'array'), val = val.array;
    else, f=fieldnames(val); val=val.(f{1}); end
end

audioSig = double(val);
if size(audioSig,2)>1, audioSig=audioSig(:,1); end
fs = 16000;
durationSec = length(audioSig)/fs;
audioSizeKB = (length(audioSig) * 8) / 1024; % 8 bytes per double

fprintf('Target Audio:  Row %d (%.2f seconds long, %.2f KB raw data)\n', rowIdx, durationSec, audioSizeKB);

% --- 5. RUN BENCHMARK ---
fprintf('\nProcessing ONE audio file... please wait... \n');
tic; 

% A. Standard Features
afeFeats = [];
if any([afeValues{:}])
    aFE = audioFeatureExtractor('SampleRate',fs, 'Window',hamming(round(0.03*fs),'periodic'), 'OverlapLength',round(0.02*fs));
    set(aFE, afeConfig);
    afeFeats = extract(aFE, audioSig);
end

% B. Custom Features (PRONY)
customFeats = [];
numFrames = max(size(afeFeats,1), 1);
winLen = round(0.03 * fs);
hop = winLen - round(0.02 * fs);

if customConfig.dampProny || customConfig.freqProny
    prony_win_sec = 0.03;      
    all_tracks = [];
    % Only process necessary chunks to be fast but accurate for this row
    % We process the whole row to get a real "per-row" time
    
    % Simplified loop from your script
    for offset_i = 0:2
        start_samp = 1 + (offset_i * hop);
        if start_samp > length(audioSig), break; end
        audio_shifted = audioSig(start_samp:end);
        if length(audio_shifted) < round(prony_win_sec * fs), continue; end
        try
            [f_map, amp_map, d_map, t_vec, w_len, n_win] = fast_prony_tracker(audio_shifted, fs, prony_win_sec);
            % Just running the tracker is the bottleneck, we don't need to format the output for the timing test
        catch
        end
    end
end

sampleTime = toc;

% --- 6. CALCULATIONS & REPORT ---
fprintf('\n--- BENCHMARK RESULTS ---\n');
fprintf('Time for 1 Audio Row:      %.4f seconds\n', sampleTime);

% Extrapolations
estTimePerParquet = sampleTime * numRowsInFile;
totalParquetFilesEstimate = (10 * 1024) / 73; % 10GB / 73MB approx 140 files
estTimeTotal = estTimePerParquet * totalParquetFilesEstimate;

fprintf('\n--- PROJECTIONS ---\n');
fprintf('1. Standard Parquet File (73 MB / %d rows):\n', numRowsInFile);
fprintf('   -> %.2f minutes (%.2f hours)\n', estTimePerParquet/60, estTimePerParquet/3600);

fprintf('\n2. Full 10 GB Dataset (~140 files):\n');
fprintf('   -> %.2f hours (%.2f days)\n', estTimeTotal/3600, estTimeTotal/86400);

if estTimeTotal > 43200 % > 12 hours
    fprintf('\n[!] CRITICAL: Your configuration is too slow for 10GB of data.\n');
    fprintf('    Cause: Prony features on every frame.\n');
    fprintf('    Fix:   Disable customConfig.dampProny/freqProny OR use a massive cluster.\n');
end