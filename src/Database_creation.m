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

fprintf('=== Building Drone Database (Structure: droneDB.Class.inst) ===\n');
droneDB = struct();

for i = 1:length(droneFolders)
    folderName = droneFolders(i).name;
    folderPath = fullfile(sourcePath, folderName);
    audioFiles = dir(fullfile(folderPath, '*.wav'));
    
    if isempty(audioFiles)
        fprintf('empty file: %s\n', folderName);
        continue;
    end
    
    
    validFieldName = matlab.lang.makeValidName(folderName);
    fprintf('processing drone: %s (%d files)...\n', folderName, length(audioFiles));
    
  
    smallest_audio_length = inf;
    for k = 1:length(audioFiles)
        info = audioinfo(fullfile(folderPath, audioFiles(k).name));
        if info.TotalSamples < smallest_audio_length
            smallest_audio_length = info.TotalSamples;
        end
    end
    

    classInstances = struct('InstanceNumber', {}, 'Signal', {}, 'PSD', {}, 'f_vec', {}, 'fs', {}, 'smallestLength', {});
    
    for j = 1:length(audioFiles)
        [sig, fs] = audioread(fullfile(folderPath, audioFiles(j).name));
        
        
        if size(sig, 2) > 1, sig = sig(:, 1); end 
        
       
        sig_centered = sig - mean(sig);
        
        
        if length(sig_centered) > smallest_audio_length
            sig_centered = sig_centered(1:smallest_audio_length);
        end
        
        
        [pxx, f_vec] = pwelch(sig_centered, hamming(1024), 512, 1024, fs);
        pxx = pxx ./ norm(pxx);
        
        
        classInstances(j).InstanceNumber = j;
        classInstances(j).Signal = sig_centered;
        classInstances(j).PSD = pxx;
        classInstances(j).f_vec = f_vec;
        classInstances(j).fs = fs;
        classInstances(j).smallestLength = smallest_audio_length;
    end
    
   
    droneDB.(validFieldName).inst = classInstances;
end


saveFileName = fullfile(savePath, 'DroneDatabase.mat');
save(saveFileName, 'droneDB', '-v7.3'); 
fprintf('\n done!:\n%s\n', saveFileName);