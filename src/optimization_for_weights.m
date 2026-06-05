clear; clc; close all;

%load data
dbPath = 'C:\AUDIO_FOR_PYTHON_CODE\Database_Output';

trainFile = fullfile(dbPath, 'DroneDatabase.mat');
load(trainFile); 
dataBase = droneDB; 

optFile = fullfile(dbPath, 'data_optimization.mat');
load(optFile); 

class_names_list = fieldnames(optDB);
if isempty(class_names_list)
    error('No classes found in the optimization database.');
end


final_data_array = {};
file_counter = 1;

for i = 1:length(class_names_list)
    outer_class_name = class_names_list{i};
    fprintf('processing class %d out of %d: %s\n', i, length(class_names_list), outer_class_name);
    
    
    instances = optDB.(outer_class_name).inst;
    num_instances = length(instances);
    
    for j = 1:num_instances
        
        test_PSD = instances(j).PSD;
        prony_freq_arr = instances(j).Prony_freq;
        mfcc=instances(j).MFCC;

        current_file_scores = cell(1, length(class_names_list));
        
        for k = 1:length(class_names_list)
            inner_class_name = class_names_list{k};
            
            grade_ind_PSD = comp_functions.individual_PSD_grading(inner_class_name, test_PSD, dataBase);
            grade_mean_PSD = comp_functions.mean_PSD_grading(inner_class_name, test_PSD, dataBase);
            grade_prony = comp_functions.prony_grading(inner_class_name, prony_freq_arr, dataBase);
            grade_MFCC = comp_functions.individual_mfcc_grading(inner_class_name, mfcc, dataBase);
            
            current_file_scores{k} = [grade_ind_PSD, grade_mean_PSD, grade_prony, grade_MFCC];
        end
        
        
        final_data_array{file_counter, 1} = outer_class_name;
        final_data_array{file_counter, 2} = current_file_scores;
        file_counter = file_counter + 1;
    end
end

disp(['Finished processing. Total instances used for optimization: ', num2str(file_counter - 1)]);


fprintf('\n--- Sanity Check: Classes Matching ---\n');
fprintf('1. class_names_list:      ');
fprintf('[%s] ', class_names_list{:});
fprintf('\n');
unique_classes_in_data = unique(final_data_array(:, 1), 'stable');
fprintf('2. final_data_array keys: ');
fprintf('[%s] ', unique_classes_in_data{:});
fprintf('\n--------------------------------------\n\n');


w_vec = optimize_drone_weights(final_data_array, class_names_list);


save_dir = 'C:\AUDIO_FOR_PYTHON_CODE';
save_file = fullfile(save_dir, 'w_vec.mat');
save(save_file, 'w_vec');
disp(['w_vec was successfully saved to: ', save_file]);



function w_optimized = optimize_drone_weights(final_data_array, class_names)
    disp('Starting Optimization Process...');
    
    %initial guess
    w0 = [25; 25; 25; 25]; 
    
    %boundries
    lb = [0; 0; 0; 0]; 
    ub = [100; 100; 100; 100];
    
    % constrictions
    Aeq = [1, 1, 1, 1]; 
    beq = 100;
    A = []; b = []; 
    
    options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp', 'MaxFunctionEvaluations', 3000);
    objectiveFunction = @(w) compute_softmax_loss(w, final_data_array, class_names);
    
    [w_optimized, ~] = fmincon(objectiveFunction, w0, A, b, Aeq, beq, lb, ub, [], options);
    
    disp('Optimization Finished!');
    disp('Optimized Weights per Feature:');
    disp(['1. Individual PSD:  ', num2str(w_optimized(1))]);
    disp(['2. mean PSD:        ', num2str(w_optimized(2))]);
    disp(['3. Prony:           ', num2str(w_optimized(3))]);
    disp(['3. centroid:           ', num2str(w_optimized(4))]);
end

function loss = compute_softmax_loss(w, data, class_names)
    loss = 0;
    num_files = size(data, 1);
    num_classes = length(class_names);
    
    % Temperature factor to prevent Softmax saturation
    T = 1; 
    
    for i = 1:num_files
        true_class = data{i, 1};
        scores_cell = data{i, 2}; 
        true_idx = find(strcmp(class_names, true_class));
        class_scores = zeros(1, num_classes);
        
        for k = 1:num_classes
            features = scores_cell{k}; 
            class_scores(k) = features * w; 
        end
        
        % Calculate Softmax probabilities with Temperature scaling
        shifted_scores = (class_scores - max(class_scores)) * T;
        exp_scores = exp(shifted_scores);
        probs = exp_scores / sum(exp_scores);
        
        % Cross-Entropy loss
        loss = loss - log(probs(true_idx) + eps);
    end
    
    loss = loss / num_files;
end