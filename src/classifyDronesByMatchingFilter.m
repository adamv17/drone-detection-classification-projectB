
rng('shuffle');

testPath  = 'C:\AUDIO_FOR_PYTHON_CODE\test'; 
N_tests   = 1;
testFolders = dir(testPath);
testFolders = testFolders([testFolders.isdir] & ~strncmp({testFolders.name}, '.', 1));

dbPath = 'C:\AUDIO_FOR_PYTHON_CODE\Database_Output';
dbFile = fullfile(dbPath, 'DroneDatabase.mat');
%load(dbFile);
if isempty(testFolders)
    error('No subfolders found in Test paths. Please check the directories.');
end
fprintf('=== Building Drone Signatures from TRAIN folder ===\n');

droneTypes = fieldnames(droneDB);
droneAverages = struct();
templates = struct();
for i = 1:length(droneTypes)

    typeName = droneTypes{i};
    all_psds = [droneDB.(typeName).inst.PSD]; 
    
    droneAverages.(typeName).avg_psd = mean(all_psds, 2);
    droneAverages.(typeName).f_vec = droneDB.(typeName).inst(1).f_vec;
        
    % Tolerance
    smear_win_hz = 15; 
    df = droneDB.(typeName).inst(1).f_vec(2) - droneDB.(typeName).inst(1).f_vec(1);
    smear_win_bins = max(1, round(smear_win_hz / df));
    avg_sig_robust = movmax(droneAverages.(typeName).avg_psd, smear_win_bins);
    
    templates(i).name = typeName;
    templates(i).signature = avg_sig_robust;
    templates(i).f_vec = droneDB.(typeName).inst(1).f_vec;
    fprintf('Template learned: %s\n', typeName);
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
    
    if isempty(testFiles), continue; end
    
    random_file = testFiles(randi(length(testFiles))).name;
    [sig, fs] = audioread(fullfile(currentTestPath, random_file));
    if size(sig, 2) > 1, sig = sig(:, 1); end
    
    
    
    
    best_score = -inf;
    predicted_name = '';
    
    for t = 1:length(templates)
        % חישוב דמיון (Dot Product)
        cutting_length= droneDB.(templates(t).name).smallestLength;
        cutted_signal_per_title = sig(1:cutting_length,:);
        [pxx_test, ~] = pwelch(cutted_signal_per_title - mean(cutted_signal_per_title), hamming(1024), 512, 1024, fs);
        pxx_test = pxx_test ./ norm(pxx_test); 
        score = sum(pxx_test .* templates(t).signature)
        if score > best_score;
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

% חלק ה-Demo Plot המעודכן (ללא הניקוי הישן)
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

[pxx_demo, f_demo] = pwelch(sig - mean(sig), hamming(1024), 512, 1024, fs);
pxx_demo = pxx_demo ./ norm(pxx_demo); 

template_idx = find(strcmp({templates.name}, actual_name));
if ~isempty(template_idx)
    figure('Name', 'Matching Demo', 'Color', 'w');
    hold on;
    
    % ציור התבנית (החתימה הממוצעת והמרוחה)
    plot(templates(template_idx).f_vec, templates(template_idx).signature, ...
        'Color', [0.8 0.8 0.8], 'LineWidth', 2, 'DisplayName', 'Train Template');
    
    % ציור ה-PSD של קובץ הבדיקה הנוכחי
    plot(f_demo, pxx_demo, 'Color', [0 0.45 0.74], 'LineWidth', 1.2, 'DisplayName', 'Test File PSD');
    
    title(['Matching for: ', actual_name], 'Interpreter', 'none');
    xlabel('Frequency (Hz)');
    ylabel('Normalized Magnitude');
    xlim([0 2000]); % התמקדות בתדרים רלוונטיים
    grid on;
    legend;
end