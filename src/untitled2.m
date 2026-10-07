clear; clc; close all;

mainPath = "C:\Users\yahal\OneDrive\מסמכים\GitHub\drone-detection-classification-projectB\src\Audacity"; 
dirContents = dir(mainPath);
subFolders = dirContents([dirContents.isdir] & ~strncmp({dirContents.name}, '.', 1));

fprintf('=== Frequency Profiler: What is actually in your files? ===\n');
fprintf('%-25s | %-15s | %-15s | %-15s\n', 'Folder Name', 'Top Peak (Hz)', '2nd Peak (Hz)', '3rd Peak (Hz)');
fprintf('----------------------------------------------------------------------\n');

for i = 1:length(subFolders)
    currentFolderName = subFolders(i).name;
    currentFolderPath = fullfile(mainPath, currentFolderName);
    audioFiles = dir(fullfile(currentFolderPath, '*.wav')); 
    
    if ~isempty(audioFiles)
        % ניקח קובץ אחד מייצג מכל תיקייה לבדיקה
        filePath = fullfile(currentFolderPath, audioFiles(1).name);
        [sig, fs] = audioread(filePath);
        if size(sig, 2) > 1, sig = sig(:, 1); end
        
        % חישוב PSD נקי
        [pxx, f] = pwelch(sig - mean(sig), hamming(16384), 8192, 32768, fs);
        
        % מציאת פיקים (מעל 40Hz כדי לסנן רעשי מנוע חשמלי/DC)
        valid_idx = f > 40 & f < 3000;
        pxx_v = pxx(valid_idx);
        f_v = f(valid_idx);
        
        [peaks, locs] = findpeaks(pxx_v, f_v, 'SortStr', 'descend', 'NPeaks', 3);
        
        if length(locs) >= 3
            fprintf('%-25s | %-15.2f | %-15.2f | %-15.2f\n', currentFolderName, locs(1), locs(2), locs(3));
        else
            fprintf('%-25s | No clear peaks found\n', currentFolderName);
        end
    end
end