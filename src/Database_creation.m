clear; clc; close all;


sourcePath = 'C:\AUDIO_FOR_PYTHON_CODE\train'; 
savePath   = 'C:\AUDIO_FOR_PYTHON_CODE\Database_Output'; 


if ~exist(savePath, 'dir')
    mkdir(savePath);
end


droneFolders = dir(sourcePath);
droneFolders = droneFolders([droneFolders.isdir] & ~strncmp({droneFolders.name}, '.', 1));

if isempty(droneFolders)
    error('No subfolders found in the source path. Please check the directory.');
end

fprintf('=== Building Drone Database ===\n');


droneDB = struct();

for i = 1:length(droneFolders)
    folderName = droneFolders(i).name;
    folderPath = fullfile(sourcePath, folderName);
    audioFiles = dir(fullfile(folderPath, '*.wav'));
    
    if isempty(audioFiles)
        fprintf('Skipping empty folder: %s\n', folderName);
        continue;
    end
    
    
    validFieldName = matlab.lang.makeValidName(folderName);
    
    fprintf('Processing %s (%d files)...\n', folderName, length(audioFiles));
    
    
    classInstances = struct('InstanceNumber', {}, 'Signal', {}, 'FFT', {}, 'PSD', {}, 'f_vec', {}, 'fs', {});
    
    for j = 1:length(audioFiles)
        
        [sig, fs] = audioread(fullfile(folderPath, audioFiles(j).name));
        if size(sig, 2) > 1, sig = sig(:, 1); end % הפיכה למונו
        
        %DC offset
        sig_centered = sig - mean(sig);
        
        % normelized PSD
        [pxx, f_vec] = pwelch(sig_centered, hamming(8192), 4096, 8192, fs);
        pxx = pxx ./ norm(pxx); 
        
        % FFT
        sig_fft = fft(sig_centered);
        
       
        classInstances(j).InstanceNumber = j;
        classInstances(j).Signal = sig;
        classInstances(j).FFT = sig_fft;
        classInstances(j).PSD = pxx;
        classInstances(j).f_vec = f_vec;
        classInstances(j).fs = fs;
    end
    
    
    droneDB.(validFieldName) = classInstances;
end


saveFileName = fullfile(savePath, 'DroneDatabase.mat');
fprintf('\nSaving database to: %s\n', saveFileName);

save(saveFileName, 'droneDB', '-v7.3'); 

fprintf('Done! Database saved successfully.\n');

