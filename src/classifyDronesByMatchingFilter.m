clear; clc; close all;
rng('shuffle');

trainPath = 'C:\AUDIO_FOR_PYTHON_CODE\train';
testPath  = 'C:\AUDIO_FOR_PYTHON_CODE\test'; 
N_tests   = 1000;



trainFolders = dir(trainPath);
trainFolders = trainFolders([trainFolders.isdir] & ~strncmp({trainFolders.name}, '.', 1));

testFolders = dir(testPath);
testFolders = testFolders([testFolders.isdir] & ~strncmp({testFolders.name}, '.', 1));

if isempty(trainFolders) || isempty(testFolders)
    error('No subfolders found in Train or Test paths. Please check the directories.');
end


fprintf('=== Building Drone Signatures from TRAIN folder ===\n');
templates = struct();
for i = 1:length(trainFolders)
    folderName = trainFolders(i).name;
    folderPath = fullfile(trainPath, folderName);
    audioFiles = dir(fullfile(folderPath, '*.wav'));
    
    if isempty(audioFiles)
        fprintf('Skipping empty folder: %s\n', folderName);
        continue;
    end
    
    all_psds = [];
    
    for j = 1:length(audioFiles)
        [sig, fs] = audioread(fullfile(folderPath, audioFiles(j).name));
        if size(sig, 2) > 1, sig = sig(:, 1); end % Mono
        
        %PSD
        [pxx, f_vec] = pwelch(sig - mean(sig), hamming(8192), 4096, 8192, fs);
        
        %noise
        %pxx_log = 10*log10(pxx + eps);
        %pxx_clean = pxx_log - movmedian(pxx_log, 200);
        %pxx_clean(pxx_clean < 0) = 0;
        
        all_psds = [all_psds, pxx];
    end
    
    
    avg_sig = mean(all_psds, 2);
    
    % Tolerance
    smear_win_hz = 15; 
    df = f_vec(2) - f_vec(1); 
    smear_win_bins = round(smear_win_hz / df);
    avg_sig_robust = movmax(avg_sig, smear_win_bins); 
    
    templates(i).name = folderName;
    templates(i).signature = avg_sig_robust;
    templates(i).f_vec = f_vec;
    fprintf('Template learned: %s\n', folderName);
end


templates = templates(~cellfun(@isempty, {templates.name}));


correct_matches = 0;
fprintf('\n=== Running %d Tests from TEST folder ===\n', N_tests);
tic; 
for k = 1:N_tests
    
    actual_test_idx = randi(length(testFolders));
    actual_name = testFolders(actual_test_idx).name;
    
    currentTestPath = fullfile(testPath, actual_name);
    testFiles = dir(fullfile(currentTestPath, '*.wav'));
    
    
    if isempty(testFiles)
        continue;
    end
    
    
    random_file = testFiles(randi(length(testFiles))).name;
    [sig, fs] = audioread(fullfile(currentTestPath, random_file));
    if size(sig, 2) > 1, sig = sig(:, 1); end
    
    
    [pxx_test, ~] = pwelch(sig - mean(sig), hamming(8192), 4096, 8192, fs);
    %test_clean = 10*log10(pxx_test + eps);
    %test_clean = test_clean - movmedian(test_clean, 200);
    %test_clean(test_clean < 0) = 0;
    
    
    best_score = -inf;
    predicted_name = '';
    
    for t = 1:length(templates)
        score = sum(pxx_test .* templates(t).signature);
        if score > best_score
            best_score = score;
            predicted_name = templates(t).name; 
        end
    end
    
    
    if strcmp(actual_name, predicted_name)
        correct_matches = correct_matches + 1;
    end
    
    if mod(k, 100) == 0
        fprintf('Progress: %d/%d | Current Accuracy: %.2f%%\n', k, N_tests, (correct_matches/k)*100);
    end
end
fprintf('\nFINAL ACCURACY (%d runs): %.2f%% (Time: %.1fs)\n', N_tests, (correct_matches/N_tests)*100, toc);


fprintf('\n=== Generating Demo Plot ===\n');


while true
    actual_test_idx = randi(length(testFolders));
    actual_name = testFolders(actual_test_idx).name;
    currentTestPath = fullfile(testPath, actual_name);
    testFiles = dir(fullfile(currentTestPath, '*.wav'));
    if ~isempty(testFiles), break; end
end

demoFileName = testFiles(randi(length(testFiles))).name;
[sig, fs] = audioread(fullfile(currentTestPath, demoFileName));
if size(sig, 2) > 1, sig = sig(:, 1); end

[pxx_demo, f_demo] = pwelch(sig - mean(sig), hamming(8192), 4096, 8192, fs);
demo_clean = 10*log10(pxx_demo + eps);
demo_clean = demo_clean - movmedian(demo_clean, 200);
demo_clean(demo_clean < 0) = 0;

template_idx = find(strcmp({templates.name}, actual_name));

if ~isempty(template_idx)
    % יצירת הגרף
    figure('Name', 'Tolerance Demo: Train Template vs Test Signal', 'Color', 'w', 'Position', [100 100 1000 600]);
    hold on;
    
    % 1. ציור התבנית המרוחה מתיקיית האימון (ה"גבעות")
    plot(templates(template_idx).f_vec, templates(template_idx).signature, ...
        'Color', [0.8 0.8 0.8], 'LineWidth', 2, 'DisplayName', 'Train Template (Robust)');
    fill(templates(template_idx).f_vec, templates(template_idx).signature, ...
        [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % 2. ציור הפיקים של קובץ הבדיקה מתיקיית הטסט (ה"שן")
    plot(f_demo, demo_clean, 'Color', [0 0.45 0.74], 'LineWidth', 1.2, 'DisplayName', 'Test File Peaks');
    
    % הגדרות הגרף
    title({'Train vs Test Matching'; ...
        ['Test File: ', demoFileName]; ['True Class: ', actual_name]}, 'Interpreter', 'none');
    xlabel('Frequency (Hz)');
    ylabel('Normalized Energy (Whitened dB)');
    xlim([40 2000]); 
    grid on;
    legend('Location', 'northeast', 'FontSize', 11);
    
    fprintf('Demo plot generated for %s.\n', actual_name);
else
    fprintf('Could not generate demo plot: Class %s exists in Test but not in Train.\n', actual_name);
end