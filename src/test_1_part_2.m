clear; clc; close all;
rng('shuffle');

% הגדרת הנתיב (תוודא שהוא נכון אצלך)
mainPath = "C:\Users\yahal\OneDrive\מסמכים\GitHub\drone-detection-classification-projectB\src\Audacity"; 
dirContents = dir(mainPath);
subFolders = dirContents([dirContents.isdir] & ~strncmp({dirContents.name}, '.', 1));

if isempty(subFolders)
    error('No subfolders found in the specified path.');
end

% --- שלב 1: בניית תבניות עם "מרחב תמרון" (Tolerance Phase) ---
fprintf('=== Building Drone Signatures with Tolerance ===\n');
templates = struct();
for i = 1:length(subFolders)
    folderName = subFolders(i).name;
    folderPath = fullfile(mainPath, folderName);
    audioFiles = dir(fullfile(folderPath, '*.wav'));
    
    if isempty(audioFiles)
        fprintf('Skipping empty folder: %s\n', folderName);
        continue;
    end
    
    all_psds = [];
    % לומדים מכל הקבצים בתיקייה כדי ליצור חתימה ממוצעת
    for j = 1:length(audioFiles)
        [sig, fs] = audioread(fullfile(folderPath, audioFiles(j).name));
        if size(sig, 2) > 1, sig = sig(:, 1); end % Mono
        
        % חישוב PSD
        [pxx, f_vec] = pwelch(sig - mean(sig), hamming(8192), 4096, 8192, fs);
        
        % ניקוי רעשים: Whitening (משאיר רק את הפיקים הצרים)
        pxx_log = 10*log10(pxx + eps);
        pxx_clean = pxx_log - movmedian(pxx_log, 200);
        pxx_clean(pxx_clean < 0) = 0; % נתמקד רק בפיקים שבולטים
        
        all_psds = [all_psds, pxx_clean];
    end
    
    % ממוצע חתימות
    avg_sig = mean(all_psds, 2);
    
    % --- הוספת Tolerance (מרחב תמרון) ---
    % נמרח כל פיק דק ל"גבעה" ברוחב של בערך 15Hz
    % זה מאפשר לסל"ד של המנוע לסטות קצת ועדיין להיקלט.
    smear_win_hz = 15; 
    df = f_vec(2) - f_vec(1); % רזולוציית תדר
    smear_win_bins = round(smear_win_hz / df);
    avg_sig_robust = movmax(avg_sig, smear_win_bins); 
    
    templates(i).name = folderName;
    templates(i).signature = avg_sig_robust;
    templates(i).f_vec = f_vec;
    fprintf('Template optimized: %s\n', folderName);
end

% הסרת תבניות ריקות (אם היו)
templates = templates(~cellfun(@isempty, {templates.name}));

% --- שלב 2: 1000 הרצות רנדומליות (Testing Phase) ---
n_iterations = 1000;
correct_matches = 0;

fprintf('\n=== Running 1000 Robust Tests ===\n');
tic; % מדידת זמן
for k = 1:n_iterations
    % הגרלת תיקייה
    actual_idx = randi(length(templates));
    actual_name = templates(actual_idx).name;
    
    % קריאת קובץ רנדומלי (לא למטרת הדגמה)
    currentFolderPath = fullfile(mainPath, actual_name);
    audioFiles = dir(fullfile(currentFolderPath, '*.wav'));
    [sig, fs] = audioread(fullfile(currentFolderPath, audioFiles(randi(length(audioFiles))).name));
    if size(sig, 2) > 1, sig = sig(:, 1); end
    
    % עיבוד הקלטת המבחן (ניקוי בלי מריחה)
    [pxx_test, ~] = pwelch(sig - mean(sig), hamming(8192), 4096, 8192, fs);
    test_clean = 10*log10(pxx_test + eps);
    test_clean = test_clean - movmedian(test_clean, 200);
    test_clean(test_clean < 0) = 0;
    
    % Matching: Dot Product (מכפלה פנימית)
    best_score = -inf;
    predicted_idx = 1;
    for t = 1:length(templates)
        % הציון הוא סכום המכפלות של הפיקים בבדיקה מול ה"גבעות" בתבנית
        score = sum(test_clean .* templates(t).signature);
        if score > best_score
            best_score = score;
            predicted_idx = t;
        end
    end
    
    if actual_idx == predicted_idx
        correct_matches = correct_matches + 1;
    end
    
    if mod(k, 100) == 0
        fprintf('Progress: %d/1000 | Current Accuracy: %.2f%%\n', k, (correct_matches/k)*100);
    end
end
fprintf('\nFINAL ACCURACY (1000 runs): %.2f%% (Time: %.1fs)\n', (correct_matches/n_iterations)*100, toc);

% --- שלב 3: הדגמה ויזואלית של "חיבוק הפיקים" ---
fprintf('\n=== Generating Demo Plot ===\n');

% הגרלת קובץ אחד נוסף להדגמה
actual_idx = randi(length(templates));
actual_name = templates(actual_idx).name;
currentFolderPath = fullfile(mainPath, actual_name);
audioFiles = dir(fullfile(currentFolderPath, '*.wav'));
demoFileName = audioFiles(randi(length(audioFiles))).name;
[sig, fs] = audioread(fullfile(currentFolderPath, demoFileName));
if size(sig, 2) > 1, sig = sig(:, 1); end

% עיבוד הקובץ להדגמה
[pxx_demo, f_demo] = pwelch(sig - mean(sig), hamming(8192), 4096, 8192, fs);
demo_clean = 10*log10(pxx_demo + eps);
demo_clean = demo_clean - movmedian(demo_clean, 200);
demo_clean(demo_clean < 0) = 0;

% יצירת הגרף
figure('Name', 'Tolerance Demo: How Peaks "Hug"', 'Color', 'w', 'Position', [100 100 1000 600]);
hold on;

% 1. ציור התבנית המרוחה (ה"גבעות")
plot(templates(actual_idx).f_vec, templates(actual_idx).signature, ...
    'Color', [0.8 0.8 0.8], 'LineWidth', 2, 'DisplayName', 'Robust Template (Gaps Filled)');
fill(templates(actual_idx).f_vec, templates(actual_idx).signature, ...
    [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 2. ציור הפיקים של קובץ הבדיקה (ה"שן")
plot(f_demo, demo_clean, 'Color', [0 0.45 0.74], 'LineWidth', 1.2, 'DisplayName', 'Test File Peaks');

% הגדרות הגרף
title({'Spectral Matching with Tolerance'; ...
    ['File: ', demoFileName, ' (True Class: ', actual_name, ')']}, 'Interpreter', 'none');
xlabel('Frequency (Hz)');
ylabel('Normalized Energy (Whitened dB)');
xlim([40 2000]); % התמקדות בטווח ה-BPF וההרמוניות
grid on;
legend('Location', 'northeast', 'FontSize', 11);

% הוספת טקסט שמסביר מה רואים
dim = [.15 .6 .3 .3];
str = {'Example of Tolerance:'; ...
       'The gray areas are "catchment zones"'; ...
       'around the template peaks. Even if'; ...
       'the blue test peak shifts slightly'; ...
       '(due to RPM change), it still lands'; ...
       'inside the gray zone, generating'; ...
       'a high score.'};
annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1);

fprintf('Demo plot generated for %s.\n', actual_name);