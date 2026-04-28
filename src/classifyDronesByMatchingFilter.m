
rng('shuffle');
testPath  = 'C:\AUDIO_FOR_PYTHON_CODE\test'; 
N   = 100;
testFolders = dir(testPath);
testFolders = testFolders([testFolders.isdir] & ~strncmp({testFolders.name}, '.', 1));
dbPath = 'C:\AUDIO_FOR_PYTHON_CODE\Database_Output';
dbFile = fullfile(dbPath, 'DroneDatabase.mat');

% טעינת הקובץ - המשתנה שנשמר בתוכו הוא droneDB
load(dbFile);

% השמה למשתנה בשם dataBase כדי שיתאים לפונקציות הדירוג שלך
dataBase = droneDB;

Class_names = fieldnames(dataBase);

if isempty(testFolders)
    error('No subfolders found in Test paths. Please check the directories.');
end
fprintf('=== Building Drone Signatures from TRAIN folder ===\n');


successes=0;
estimate = 'Unknown';
for i=1:N
    fprintf('working on %d\n', i);
    randomFolderIdx = randi(length(testFolders));
    selectedFolder = testFolders(randomFolderIdx).name;
    selectedPath = fullfile(testPath, selectedFolder);
    
    % 2. בחירת קובץ רנדומלי מהתיקייה שנבחרה
    testFiles = dir(fullfile(selectedPath, '*.wav'));
    if isempty(testFiles)
        error('לא נמצאו קבצי wav בתיקייה: %s', selectedFolder);
    end
    randomFileIdx = randi(length(testFiles));
    randomDroneFile = fullfile(selectedPath, testFiles(randomFileIdx).name);
    
    % 3. הרצת הבדיקה
    fprintf('--- Testing Random Drone ---\n');
    fprintf('Selected from folder: %s\n', selectedFolder);
    fprintf('File name: %s\n', testFiles(randomFileIdx).name);
    
    [estimate] = find_drone_type(randomDroneFile, dataBase, Class_names);
    fprintf('Estimate: [%s], Actual: [%s]\n', estimate, selectedFolder);
    if strcmp(estimate, selectedFolder)
        successes=successes+1;
    end
end

fprintf('success rate: %.2f%%\n', (successes * 100) / N);







function [estimate_drone_type] = find_drone_type(drone_recording, dataBase, Class_names)
    [test_sig, fs] = audioread(drone_recording);
    if size(test_sig, 2) > 1, test_sig = test_sig(:, 1); end
    w=optimization_for_weights.optimize_weights(dataBase);
    w= [0.25,0.25,0.25,0.25];
    
    score_vec=[0,0,0,0];
    for i=1: length(Class_names)
        
        if length(test_sig)<dataBase.(Class_names{i}).inst(1).smallestLength
            current_sig = test_sig;
            current_sig(dataBase.(Class_names{i}).inst(1).smallestLength, 1) = 0;
        else
            current_sig=test_sig(1:dataBase.(Class_names{i}).inst(1).smallestLength);
        end
        [test_PSD, ~] = pwelch(current_sig - mean(current_sig), hamming(1024), 512, 1024, fs);
        test_PSD = test_PSD ./ norm(test_PSD); 
        grade_ind_PSD=comp_functions.individual_PSD_grading(Class_names{i},test_PSD,dataBase);
        grade_ind_Time=comp_functions.individual_Time_grading(Class_names{i},current_sig,dataBase);
        grade_mean_PSD=comp_functions.mean_PSD_grading(Class_names{i},test_PSD,dataBase);
        grade_mean_Time=comp_functions.mean_Time_grading(Class_names{i},current_sig,dataBase);
        current_score_vec = w.*[grade_ind_PSD,grade_ind_Time,grade_mean_PSD,grade_mean_Time];   
        better_metrics_count = sum(current_score_vec > score_vec);
        if better_metrics_count >= 3
            estimate_drone_type = Class_names{i};
            score_vec = current_score_vec;
        end
    end

end





