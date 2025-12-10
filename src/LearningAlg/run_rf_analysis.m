%% run_rf_analysis.m
% ---------------------------------------------------------
% PURPOSE: 
% 1. Trains Random Forest on 'TrainTable'.
% 2. Validates on 'TestTable'.
% 3. Calculates Detailed Metrics (Precision, Recall, F1).
% 4. Prints filenames of mistakes.
% 5. Visualizes Feature Importance & Tree Structure.
% ---------------------------------------------------------

if ~exist('TrainTable', 'var') || ~exist('TestTable', 'var')
    error('Data not found! Please run "learner_setup.m" first.');
end

fprintf('Loading data from workspace...\n');

% --- 1. DATA PREPARATION ---
% Identify feature columns by excluding metadata
nonFeatureCols = {'Label', 'Filename'};
featureNames = setdiff(TrainTable.Properties.VariableNames, nonFeatureCols, 'stable');

% Extract Matrix X (Features Only) and Y (Labels)
X_Train = TrainTable{:, featureNames}; 
Y_Train = TrainTable.Label;

fprintf('Training on %d features...\n', length(featureNames));

%% 2. TRAIN RANDOM FOREST
fprintf('Training Random Forest (50 Trees)...\n');
numTrees = 50;

rf_model = TreeBagger(numTrees, X_Train, Y_Train, ...
    'Method', 'classification', ...
    'OOBPredictorImportance', 'on', ... 
    'OOBPrediction', 'on', ...
    'PredictorNames', featureNames); 

fprintf('Training Complete.\n');

%% 3. TEST SET VALIDATION
fprintf('\nEvaluating on Test Set...\n');

% Extract Test Data safely using feature names
X_Test = TestTable{:, featureNames};
Y_Test = TestTable.Label;
Filenames_Test = TestTable.Filename;

% Predict
[pred_labels, scores] = predict(rf_model, X_Test);
pred_labels = categorical(pred_labels);

% --- DETAILED METRICS CALCULATION ---
% Calculate Confusion Matrix components
C = confusionmat(Y_Test, pred_labels);
% Note: Confusion Matrix C(i,j) is count of True class i classified as j
% Assuming Class 1 = DRONE, Class 2 = OTHER (Alphabetical order usually)
classes = categories(Y_Test);
droneIdx = find(strcmpi(classes, 'DRONE'));

if isempty(droneIdx)
    warning('Could not find "DRONE" class. Check label names.');
    droneIdx = 1; 
end

% Extract TP, TN, FP, FN
% This logic generalizes for multi-class but we focus on binary DRONE vs OTHER
TP = C(droneIdx, droneIdx);
FN = sum(C(droneIdx, :)) - TP;
FP = sum(C(:, droneIdx)) - TP;
TN = sum(C(:)) - (TP + FP + FN);

% Metrics
accuracy = (TP + TN) / sum(C(:));
precision = TP / (TP + FP);
recall = TP / (TP + FN); % Sensitivity
f1_score = 2 * (precision * recall) / (precision + recall);
fpr = FP / (FP + TN); % False Positive Rate

% Handle NaN if division by zero occurs
if isnan(precision), precision = 0; end
if isnan(f1_score), f1_score = 0; end

fprintf('\n------------------------------------------------\n');
fprintf('PERFORMANCE METRICS (Target Class: DRONE)\n');
fprintf('------------------------------------------------\n');
fprintf('Accuracy:            %.2f%%\n', accuracy * 100);
fprintf('Precision:           %.2f%%\n', precision * 100);
fprintf('Recall (Sensitivity):%.2f%%\n', recall * 100);
fprintf('F1-Score:            %.2f%%\n', f1_score * 100);
fprintf('False Positive Rate: %.2f%%\n', fpr * 100);
fprintf('------------------------------------------------\n');

%% 4. VISUALIZATIONS
figure('Position', [100, 100, 1400, 600], 'Color', 'w');

% Plot A: Confusion Matrix
subplot(1, 2, 1);
cm_chart = confusionchart(Y_Test, pred_labels);
cm_chart.Title = {['Test Accuracy: ' num2str(accuracy*100, '%.2f') '%'], ...
                  ['F1-Score: ' num2str(f1_score, '%.2f')]};

% Plot B: Feature Importance
subplot(1, 2, 2);
importance_scores = rf_model.OOBPermutedPredictorDeltaError;
[sorted_scores, sorted_idx] = sort(importance_scores, 'descend');

topN = min(20, length(featureNames)); 
barh(sorted_scores(1:topN));
set(gca, 'YTick', 1:topN, ...
    'YTickLabel', featureNames(sorted_idx(1:topN)), ...
    'TickLabelInterpreter', 'none'); 
xlabel('Importance (Delta Error)'); 
title(['Top ' num2str(topN) ' Features']);
grid on;

%% 5. DETAILED MISTAKE PRINTER (Filename + Timestamps)
fprintf('\n------------------------------------------------\n');
fprintf('          MISCLASSIFIED SEGMENTS DETAILED       \n');
fprintf('------------------------------------------------\n');

mistakeIdx = find(pred_labels ~= Y_Test);

if isempty(mistakeIdx)
    disp('No mistakes found in the test set!');
else
    % Get unique filenames that contain at least one mistake
    uniqueMistakes = unique(Filenames_Test(mistakeIdx));
    
    % Define Windowing Parameters used in extraction (Must match learner_setup)
    % Standard values: 30ms window, 10ms hop (0.01s)
    fs_nominal = 44100;
    hop_samples = round(0.01 * fs_nominal); 
    hop_time = hop_samples / fs_nominal; % Should be approx 0.01s

    for i = 1:length(uniqueMistakes)
        fname = uniqueMistakes(i);
        
        % 1. Find all rows for this file
        fileGlobalIdx = find(Filenames_Test == fname);
        
        % 2. Find which of these rows were mistakes
        % Intersect global file indices with global mistake indices
        fileMistakeIdx = intersect(fileGlobalIdx, mistakeIdx);
        
        % 3. Calculate Timestamps
        % We need the LOCAL frame index (1st frame of file, 2nd frame...)
        % Since 'fileGlobalIdx' is sequential, we can map:
        % Local Index = (Global Index - First Index of File)
        firstFrameIdx = fileGlobalIdx(1);
        localMistakeIdx = fileMistakeIdx - firstFrameIdx; 
        
        % Time = LocalIndex * HopTime
        mistakeTimes = localMistakeIdx * hop_time;
        
        % 4. Format the Output
        % Convert list of times [0.01, 0.02, 0.03, 0.50] into readable ranges
        % or a compact string
        
        % Quick labeling check
        actual = string(Y_Test(fileMistakeIdx(1)));
        predicted = string(pred_labels(fileMistakeIdx(1)));
        
        fprintf('File: %s\n', fname);
        fprintf('   Type: Actual [%s] vs Predicted [%s]\n', actual, predicted);
        
        % Print in groups of 10 timestamps to keep it readable
        if length(mistakeTimes) > 20
             fprintf('   Mistakes at: %.2fs to %.2fs (Total %d frames)\n', ...
                 min(mistakeTimes), max(mistakeTimes), length(mistakeTimes));
        else
             fprintf('   Mistakes at: %s seconds\n', num2str(mistakeTimes', '%.2f '));
        end
        fprintf('------------------------------------------------\n');
    end
end

%% 6. VISUALIZE A SINGLE TREE
fprintf('\nVisualizing Tree #1...\n');
first_tree = rf_model.Trees{1}; 

figure('Name', 'Decision Tree Visualization', 'Color', 'w');
view(first_tree, 'Mode', 'graph'); 
title('Visualizing Tree #1 of the Forest');

%% Section 7: Unified Performance Dashboard (Success, Failure & ROC)
% ---------------------------------------------------------
% PURPOSE: 
% 1. Visualizes a "Best Case" file (High Accuracy).
% 2. Visualizes a "Worst Case" file (Mistakes/Low Accuracy).
% 3. Displays Global ROC Curves (Raw vs. Filtered).
% ---------------------------------------------------------

fprintf('\nGenerating Unified Dashboard...\n');

% --- 1. SETUP & DATA PREPARATION ---

% A. Identify the correct column for 'DRONE'
droneColIdx = find(strcmp(rf_model.ClassNames, 'DRONE'));
if isempty(droneColIdx)
    error('Class "DRONE" not found in model! Check label names.');
end
fprintf('Target Class "DRONE" found at Column Index: %d\n', droneColIdx);

% B. Apply Smoothing to ALL Test Files (for Global ROC)
smoothed_scores = zeros(size(scores, 1), 1);
raw_drone_scores = scores(:, droneColIdx); 
unique_files = unique(Filenames_Test);
window_size = 20; 

for i = 1:length(unique_files)
    f_idx = strcmp(Filenames_Test, unique_files{i});
    smoothed_scores(f_idx) = movmean(raw_drone_scores(f_idx), window_size);
end

% --- 2. SELECT REPRESENTATIVE FILES ---

file_accuracies = zeros(length(unique_files), 1);
for i = 1:length(unique_files)
    f_idx = strcmp(Filenames_Test, unique_files{i});
    file_accuracies(i) = mean(pred_labels(f_idx) == Y_Test(f_idx));
end

% A. Find "Success" File
is_drone_file = contains(string(unique_files), 'DRONE', 'IgnoreCase', true);
if any(is_drone_file)
    [~, best_idx] = max(file_accuracies .* is_drone_file); 
else
    [~, best_idx] = max(file_accuracies);
end
success_file = unique_files{best_idx};

% B. Find "Failure" File
[~, worst_idx] = min(file_accuracies);
failure_file = unique_files{worst_idx};

fprintf('Success Example: %s (Acc: %.1f%%)\n', success_file, file_accuracies(best_idx)*100);
fprintf('Failure Example: %s (Acc: %.1f%%)\n', failure_file, file_accuracies(worst_idx)*100);

% Helper to extract plotting data
get_plot_data = @(fname) deal( ...
    raw_drone_scores(strcmp(Filenames_Test, fname)), ...
    smoothed_scores(strcmp(Filenames_Test, fname)), ...
    double(Y_Test(strcmp(Filenames_Test, fname)) == 'DRONE') ...
);

[raw_good, smooth_good, truth_good] = get_plot_data(success_file);
[raw_bad, smooth_bad, truth_bad]    = get_plot_data(failure_file);

% --- 3. DASHBOARD VISUALIZATION ---
fig7 = figure('Name', 'Algorithm Performance Dashboard', 'Color', 'w', 'Position', [50, 50, 1400, 800]);

% Create 2x2 Layout
tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'Algorithm Validation: Local Examples & Global ROC', 'FontSize', 16, 'FontWeight', 'bold');

% --- PANEL 1 (Top Left): SUCCESS EXAMPLE ---
% Explicitly place at Index 1 (Row 1, Col 1)
nexttile(1); 
hold on;
area(truth_good, 'FaceColor', [0.9 0.95 0.9], 'EdgeColor', 'none', 'DisplayName', 'Ground Truth (Drone)');
plot(raw_good, 'Color', [0.7 0.7 0.7], 'LineStyle', ':', 'DisplayName', 'Raw Confidence');
plot(smooth_good, 'Color', '#77AC30', 'LineWidth', 2, 'DisplayName', 'Filtered Signal'); 
yline(0.5, '--k', 'Threshold', 'HandleVisibility', 'off');
title(['SUCCESS: ' char(success_file)], 'Interpreter', 'none');
ylabel('Drone Probability');
ylim([-0.05 1.05]);
grid on; legend('Location', 'southwest');

% --- PANEL 2 (Bottom Left): FAILURE EXAMPLE ---
% Explicitly place at Index 3 (Row 2, Col 1)
nexttile(3); 
hold on;
area(truth_bad, 'FaceColor', [0.95 0.9 0.9], 'EdgeColor', 'none', 'DisplayName', 'Ground Truth (Drone)');
plot(raw_bad, 'Color', [0.7 0.7 0.7], 'LineStyle', ':', 'DisplayName', 'Raw Confidence');
plot(smooth_bad, 'Color', '#D95319', 'LineWidth', 2, 'DisplayName', 'Filtered Signal'); 
yline(0.5, '--k', 'Threshold', 'HandleVisibility', 'off');
title(['FAILURE: ' char(failure_file)], 'Interpreter', 'none');
xlabel('Frame Index'); ylabel('Drone Probability');
ylim([-0.05 1.05]);
grid on;

% --- PANEL 3 (Right Side): ROC CURVES ---
% Explicitly place at Index 2 (Top Right) and span [2 Rows, 1 Col]
nexttile(2, [2 1]); 
hold on;

[fpr_raw, tpr_raw, ~, auc_raw] = perfcurve(Y_Test, raw_drone_scores, 'DRONE');
[fpr_filt, tpr_filt, ~, auc_filt] = perfcurve(Y_Test, smoothed_scores, 'DRONE');

plot(fpr_raw, tpr_raw, 'Color', [0.6 0.6 0.6], 'LineStyle', '--', 'LineWidth', 1.5);
plot(fpr_filt, tpr_filt, 'Color', '#0072BD', 'LineWidth', 2.5);
plot([0 1], [0 1], 'k:'); 

grid on; axis square;
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('Global Model Performance (ROC)');
legend({['Raw (AUC: ' num2str(auc_raw, '%.2f') ')'], ...
        ['Filtered (AUC: ' num2str(auc_filt, '%.2f') ')']}, ...
        'Location', 'southeast', 'FontSize', 10);

%% Section 8: MFCC Forensic Analysis (Gain/Spectrum Mismatch)
% ---------------------------------------------------------
% PURPOSE: 
% 1. Compares the "Fingerprint" (Mean MFCCs) of Training Data vs File 29.
% 2. Checks if the failure is due to Loudness (MFCC_1) or Timbre (MFCC_2+).
% ---------------------------------------------------------

fprintf('\nRunning MFCC Diagnostics...\n');

% --- 1. SETUP ---
target_file = "DRONE_029.wav"; 
problem_mask = strcmp(TestTable.Filename, target_file);

if sum(problem_mask) == 0
    warning('File %s not found. Using worst file.', target_file);
    problem_mask = strcmp(TestTable.Filename, failure_file);
end

% Identify all MFCC columns automatically
mfcc_cols = contains(featureNames, 'mfcc', 'IgnoreCase', true);
mfcc_names = featureNames(mfcc_cols);

if sum(mfcc_cols) == 0
    error('No MFCC features found in table! Check feature names.');
end

% --- 2. CALCULATE FINGERPRINTS ---

% A. The "Ideal Drone" (From Training Data)
train_drone_mask = (TrainTable.Label == 'DRONE');
mean_train_drone = mean(TrainTable{train_drone_mask, mfcc_cols});
std_train_drone  = std(TrainTable{train_drone_mask, mfcc_cols});

% B. The "Confusing Background" (From Training Data)
train_bg_mask = (TrainTable.Label ~= 'DRONE');
mean_train_bg = mean(TrainTable{train_bg_mask, mfcc_cols});

% C. The "Problem File" (File 29)
mean_problem = mean(TestTable{problem_mask, mfcc_cols});

% --- 3. VISUALIZATION ---
fig8 = figure('Name', 'MFCC Fingerprint Analysis', 'Color', 'w', 'Position', [100, 100, 1000, 600]);

% Plot 1: The Fingerprint Comparison
subplot(2, 1, 1);
hold on;

% Plot Training Drone Range (Blue Shading)
x_axis = 1:length(mfcc_names);
fill([x_axis fliplr(x_axis)], ...
     [mean_train_drone-std_train_drone fliplr(mean_train_drone+std_train_drone)], ...
     'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Training Drone Range (1 StdDev)');

% Plot Lines
plot(x_axis, mean_train_drone, 'b--o', 'LineWidth', 1.5, 'DisplayName', 'Avg Training Drone');
plot(x_axis, mean_train_bg, 'k:', 'LineWidth', 1, 'DisplayName', 'Avg Background');
plot(x_axis, mean_problem, 'Color', '#D95319', 'LineWidth', 3, 'Marker', 's', 'DisplayName', 'FILE 29 (Problem)');

ylabel('Feature Value');
title('Why the Model is Confused: Feature Mismatch');
xticks(x_axis);
xticklabels(strrep(mfcc_names, '_', '\_')); % Escape underscores
xtickangle(45);
grid on; legend('Location', 'best');

% Plot 2: Distance Metric (Euclidean)
subplot(2, 1, 2);
dist_to_drone = norm(mean_problem - mean_train_drone);
dist_to_bg = norm(mean_problem - mean_train_bg);

barh([1, 2], [dist_to_drone, dist_to_bg]);
yticks([1, 2]);
yticklabels({'Distance to DRONE Class', 'Distance to BACKGROUND Class'});
xlabel('Euclidean Distance (Lower is Better)');
title(['Classification Logic: The Model thinks File 29 is ' ...
       term(dist_to_bg < dist_to_drone, 'BACKGROUND', 'UNKNOWN')]);

% Helper for title
function s = term(cond, a, b), if cond, s=a; else, s=b; end; end