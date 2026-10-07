clear; clc; close all;
sourcePath = 'C:\AUDIO_FOR_PYTHON_CODE\train'; 
savePath   = 'C:\AUDIO_FOR_PYTHON_CODE\Database_Output'; 
if ~exist(savePath, 'dir')
    mkdir(savePath);
end
droneFolders = dir(sourcePath);
droneFolders = droneFolders([droneFolders.isdir] & ~strncmp({droneFolders.name}, '.', 1));
if isempty(droneFolders)
    error('no directories in path');
end
fprintf('=== Building Drone Databases (50%% Train / 40%% Test / 10%% Optimization) ===\n');

droneDB = struct();
testDB  = struct();
optDB   = struct();

% upper limit to files length to prevent fft and prony on very long files
MAX_DURATION_SEC = 1.5; 

for i = 1:length(droneFolders)
    folderName = droneFolders(i).name;
    folderPath = fullfile(sourcePath, folderName);
    audioFiles = dir(fullfile(folderPath, '*.wav'));
    
    if isempty(audioFiles)
        fprintf('empty file: %s\n', folderName);
        continue;
    end
    
    validFieldName = matlab.lang.makeValidName(folderName);
    
    numFiles = length(audioFiles);
    shuffledIdx = randperm(numFiles);
    
    numTrain = round(0.45 * numFiles);
    numTest  = round(0.4 * numFiles);
    
    trainFiles = audioFiles(shuffledIdx(1 : numTrain));
    testFiles  = audioFiles(shuffledIdx(numTrain + 1 : numTrain + numTest));
    optFiles   = audioFiles(shuffledIdx(numTrain + numTest + 1 : end));
    
    fprintf('processing drone: %s (%d train, %d test, %d opt)...\n', ...
        folderName, length(trainFiles), length(testFiles), length(optFiles));
    
    fileSets = {trainFiles, testFiles, optFiles};
    dbNames = {'train', 'test', 'optimization'};
    
    for s = 1:3
        currFiles = fileSets{s};
        numCurrFiles = length(currFiles);
        
        if numCurrFiles == 0
            continue;
        end
        
        dummyStruct = struct('InstanceNumber', 0, 'Signal', [], 'PSD', [], 'Prony_freq', [], 'f_vec', [], 'fs', 0);
        classInstances = repmat(dummyStruct, numCurrFiles, 1);
        
        parfor j = 1:numCurrFiles
            [sig, fs] = audioread(fullfile(folderPath, currFiles(j).name));
            
            if size(sig, 2) > 1, sig = sig(:, 1); end 
            
            % --- cutting long files ---
            max_samples = round(MAX_DURATION_SEC * fs);
            if length(sig) > max_samples
                sig = sig(1:max_samples);
            end
            % ---------------------------------------
            
            sig_centered = sig - mean(sig);
            if norm(sig_centered) > 0
                sig_centered = sig_centered ./ norm(sig_centered);
            end
            
            [pxx, f_vec] = pwelch(sig_centered, hamming(1024), 512, 1024, fs);
            if norm(pxx) > 0
                pxx = pxx ./ norm(pxx);
            end

            spectral_centroid = sum(f_vec .* pxx) / sum(pxx);
            
            
            [freq_map, ~, ~, ~, ~] = prony_tracker(sig_centered, fs, 0.03);
            prony_freq_arr = mean(freq_map, 2);

            coeffs = mfcc(sig_centered, fs, 'NumCoeffs', 13);
            mfcc_vec = mean(coeffs, 1); 

            
            classInstances(j).InstanceNumber = j;
            classInstances(j).Signal = sig_centered; 
            classInstances(j).PSD = pxx;
            classInstances(j).Centroid = spectral_centroid;
            classInstances(j).Prony_freq = prony_freq_arr;
            classInstances(j).MFCC = mfcc_vec
            classInstances(j).f_vec = f_vec;
            classInstances(j).fs = fs;
        end
        
        if strcmp(dbNames{s}, 'train')
            droneDB.(validFieldName).inst = classInstances;
        elseif strcmp(dbNames{s}, 'test')
            testDB.(validFieldName).inst = classInstances;
        else
            optDB.(validFieldName).inst = classInstances;
        end
    end
end

trainSaveFileName = fullfile(savePath, 'DroneDatabase.mat');
save(trainSaveFileName, 'droneDB', '-v7.3'); 
fprintf('\n saved Train Database:\n%s\n', trainSaveFileName);

testSaveFileName = fullfile(savePath, 'test_data.mat');
save(testSaveFileName, 'testDB', '-v7.3'); 
fprintf(' saved Test Database:\n%s\n', testSaveFileName);

optSaveFileName = fullfile(savePath, 'data_optimization.mat');
save(optSaveFileName, 'optDB', '-v7.3'); 
fprintf(' saved Optimization Database:\n%s\n', optSaveFileName);

function [freq_map, amp_map, time_vec, win_len, num_of_win] = prony_tracker(signal, fs, win_len_sec)
    num_peaks = 8;               
    model_order = num_peaks * 2; 
    
    N = round(win_len_sec * fs);
    [windows, ~] = buffer(signal, N, 0, 'nodelay');
    num_wins = size(windows, 2);
    
    freq_map = zeros(num_peaks, num_wins);
    amp_map  = zeros(num_peaks, num_wins);
    
    for i = 1:num_wins
        seg = windows(:, i);
        try
            [b, a] = prony(seg, model_order, model_order);
            [r, p, ~] = residuez(b, a);
            f_hz = angle(p) * fs / (2*pi);
            mag  = abs(r);
            
            idx_valid = find(f_hz > 10); 
            curr_freqs = f_hz(idx_valid);
            curr_amps  = mag(idx_valid);
            
            [sorted_amps, sort_idx] = sort(curr_amps, 'descend');
            sorted_freqs = curr_freqs(sort_idx);
            
            count = min(length(sorted_freqs), num_peaks);
            if count > 0
                freq_map(1:count, i) = sorted_freqs(1:count);
                amp_map(1:count, i)  = sorted_amps(1:count);
            end
        catch
            continue;
        end
    end
    
    total_duration = length(signal)/fs;
    time_vec = linspace(win_len_sec/2, total_duration - win_len_sec/2, num_wins);
    win_len = win_len_sec;
    num_of_win = num_wins;
end