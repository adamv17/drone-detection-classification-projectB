clear; clc; close all;
rng('shuffle');

N = 1200; %number of tests

% load data (train group + test group)
dbPath = 'C:\AUDIO_FOR_PYTHON_CODE\Database_Output';
trainFile = fullfile(dbPath, 'DroneDatabase.mat');
testFile  = fullfile(dbPath, 'test_data.mat');
p = 'C:\AUDIO_FOR_PYTHON_CODE\w_vec.mat';

load(trainFile);
dataBase = droneDB;
load(testFile); 


% load optimized w vector
loaded_data = load(p);
w_optimized = loaded_data.w_vec;
disp('Optimized Weights:');
disp(w_optimized);





Class_names = fieldnames(testDB);
if isempty(Class_names)
    error('No classes found in the test database.');
end

successes = 0;
estimate = 'Unknown';

true_labels = cell(N, 1);
predicted_labels = cell(N, 1);

% testing on N files

for i = 1:N
    fprintf('working on %d\n', i);
    
    %random class
    randomClassIdx = randi(length(Class_names));
    selectedClass = Class_names{randomClassIdx};
    
    % random file 
    numInstances = length(testDB.(selectedClass).inst);
    randomInstIdx = randi(numInstances);
    
    test_drone = testDB.(selectedClass).inst(randomInstIdx);
    
   
    fprintf('--- Testing Random Drone ---\n');
    fprintf('Selected from class: %s\n', selectedClass);
    fprintf('Instance number: %d\n', test_drone.InstanceNumber);
    
    % estimate the drone type (class) via our algorithem
    [estimate] = find_drone_type(test_drone, dataBase, Class_names, w_optimized);

    true_labels{i} = selectedClass;
    predicted_labels{i} = estimate;
    
    fprintf('Estimate: [%s], Actual: [%s]\n', estimate, selectedClass);
    
    if strcmp(estimate, selectedClass)
        successes = successes + 1;
    end
end

fprintf('success rate: %.2f%%\n', (successes * 100) / N);

%Confusion Matrix
figure('Name', 'Test Data Confusion Matrix', 'NumberTitle', 'off');
cm = confusionchart(true_labels, predicted_labels);
cm.Title = sprintf('Drone Classification Test (Success Rate: %.2f%%)', (successes * 100) / N);
cm.RowSummary = 'row-normalized'; % Recall - מציג כמה זיהינו נכון מתוך כל מחלקה
cm.ColumnSummary = 'column-normalized'; % Precision - מציג כמה מהזיהויים שלנו היו מדויקים


% estimate class function
function [estimate_drone_type] = find_drone_type(test_drone, dataBase, Class_names, w_optimized)
    
    best_score=-inf;
    for i=1: length(Class_names)
        test_PSD=test_drone.PSD;
        prony_freq_arr=test_drone.Prony_freq;
        mfcc=test_drone.MFCC;
        
        % checking grade of 4 features
        grade_ind_PSD=comp_functions.individual_PSD_grading(Class_names{i},test_PSD,dataBase);
        grade_mean_PSD=comp_functions.mean_PSD_grading(Class_names{i},test_PSD,dataBase);
        grade_prony = comp_functions.prony_grading(Class_names{i}, prony_freq_arr, dataBase);
        grade_MFCC = comp_functions.individual_mfcc_grading(Class_names{i}, mfcc, dataBase);

        % calculating final grade for class X with w vector
        current_score = [grade_ind_PSD,grade_mean_PSD, grade_prony,grade_MFCC] * w_optimized;   
        
        if current_score >= best_score
            estimate_drone_type = Class_names{i};
            best_score = current_score;
        end

        fprintf('Class: %s | ind_PSD: %.4f | mean_PSD: %.4f | Prony: %.4f | MFCC:%.4f | dot_product: %.4f\n',  ...
            Class_names{i}, grade_ind_PSD, grade_mean_PSD, grade_prony,grade_MFCC,current_score );
    end
end


