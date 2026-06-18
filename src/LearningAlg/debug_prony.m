%% Debug_Prony.m
clear; clc;
% 1. Load one file manually
dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 
wavFiles = dir(fullfile(dataPath, '*.wav'));
if isempty(wavFiles), error('Check path'); end

% Pick the first file
fileEntry = wavFiles(1);
fullPath = fullfile(dataPath, fileEntry.name);
[audioData, fs] = audioread(fullPath);
if size(audioData, 2) > 1, audioData = audioData(:, 1); end

fprintf('Debugging file: %s\n', fileEntry.name);
fprintf('Sample Rate: %d\n', fs);

% 2. Simulate the "Fast" (30ms) chunk extraction
prony_fast_sec = 0.03;
prony_fast_samps = round(prony_fast_sec * fs);
currentCenter = round(length(audioData)/2); % Middle of file

sIdx_f = currentCenter - floor(prony_fast_samps/2);
eIdx_f = sIdx_f + prony_fast_samps - 1;
chunk_fast = audioData(sIdx_f:eIdx_f);

fprintf('Chunk size: %d samples\n', length(chunk_fast));

% 3. RUN PRONY WITHOUT TRY-CATCH
% This is where it likely breaks.
fprintf('Attempting prony_tracker on fast chunk...\n');

% Debug Call
[f_map_fast, amp_map, d_map_fast, time_vec, win_len, num_win] = prony_tracker(chunk_fast, fs, prony_fast_sec);

fprintf('Prony Tracker success. Windows found: %d\n', num_win);

% 4. RUN FEATURE EXTRACTOR
fprintf('Attempting get_features_from_prony...\n');

% Debug Call
[std_vec, d_mat_fast, f_mat_fast] = get_features_from_prony(f_map_fast, amp_map, d_map_fast, time_vec, win_len, num_win);

disp('Extraction success!');
disp('First Frequency Row:');
disp(f_mat_fast(1, 1:5));