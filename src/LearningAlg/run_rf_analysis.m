%% run_rf_analysis.m
% ---------------------------------------------------------
% PURPOSE: 
% 1. Data Preparation
% 2. Trains Random Forest on 'TrainTable'.
% 3. Validates on 'TestTable' and calculates metrics (Precision, Recall, F1).
% 4. Visualize confusion matrix and feature importance.
% 5. Prints filenames of mistakes with times.
% 6. Visualise a single tree
% 7. Visualize a dashboard of the algorithm with ROC and worst/best
% performance.
% ---------------------------------------------------------

if ~exist('TrainTable', 'var') || ~exist('TestTable', 'var')
    error('Data not found! Please run "learner_setup.m" first.');
end

fprintf('Loading data from workspace...\n');

%% --- 1. DATA PREPARATION ---
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

%% Section 8: Surgical Feature Pruning (The "Drift Fix")
% ---------------------------------------------------------
% PURPOSE: 
% 1. Compares File 29 to Training Data dimension-by-dimension.
% 2. Automatically removes features where File 29 drifts too far.
% 3. Retrains the model on ONLY the "Safe" features.
% ---------------------------------------------------------

fprintf('\nRunning Surgical Feature Pruning...\n');

% --- 1. STATISTICS SETUP ---
% Get Training Drone Stats
train_drone_mask = (TrainTable.Label == 'DRONE');
mu_train = mean(TrainTable{train_drone_mask, featureNames});
sigma_train = std(TrainTable{train_drone_mask, featureNames});

% Get Problem File Stats
target_file = "DRONE_029.wav";
idx = strcmp(TestTable.Filename, target_file);
if sum(idx) == 0, error('File 29 not found!'); end
mu_problem = mean(TestTable{idx, featureNames});

% --- 2. CALCULATE Z-SCORES (The "Drift" Metric) ---
% How many standard deviations away is File 29 from the Training Mean?
z_scores = (mu_problem - mu_train) ./ sigma_train;

% Visualization of the Drift
figure('Name', 'Drifting Feature Detection', 'Color', 'w', 'Position', [100, 100, 1000, 400]);
bar(z_scores);
xlabel('Feature Index'); ylabel('Z-Score (Drift)');
title('Which Features are betraying the model?');
xticks(1:length(featureNames));
xticklabels(strrep(featureNames, '_', '\_'));
xtickangle(45);
yline(1.5, 'r--', 'Unsafe Threshold (+1.5 sigma)');
yline(-1.5, 'r--', 'Unsafe Threshold (-1.5 sigma)');
grid on;

% --- 3. PRUNE THE FEATURES ---
% We keep only features where the drift is small (e.g., < 1.5 std devs)
safe_mask = abs(z_scores) < 1.5; 

prunedFeatureNames = featureNames(safe_mask);
removedFeatureNames = featureNames(~safe_mask);

fprintf('\n------------------------------------------------\n');
fprintf('DETECTED DRIFTING FEATURES (REMOVING):\n');
fprintf('  > %s\n', removedFeatureNames{:});
fprintf('------------------------------------------------\n');
fprintf('Retraining with %d Safe Features (originally %d)...\n', ...
    length(prunedFeatureNames), length(featureNames));

% --- 4. RETRAIN ON SAFE FEATURES ONLY ---
% We re-use the Jittered data if available, but limit columns to pruned list
if exist('X_Train_Robust', 'var')
    % Find indices of safe features
    [~, safe_cols] = ismember(prunedFeatureNames, featureNames);
    X_Safe = X_Train_Robust(:, safe_cols);
    Y_Safe = Y_Train_Robust;
else
    X_Safe = TrainTable{:, prunedFeatureNames};
    Y_Safe = TrainTable.Label;
end

rf_pruned = TreeBagger(50, X_Safe, Y_Safe, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'PredictorNames', prunedFeatureNames); % Important!

% --- 5. VERIFY THE FIX ---
[~, scores_final] = predict(rf_pruned, TestTable{idx, prunedFeatureNames});
droneCol = find(strcmp(rf_pruned.ClassNames, 'DRONE'));
final_acc = mean((scores_final(:, droneCol) > 0.5) == (TestTable.Label(idx)=='DRONE'));

fprintf('\nACCURACY AFTER PRUNING: %.1f%%\n', final_acc * 100);

if final_acc > 0.85
    fprintf('SUCCESS: Removing the drifting features solved the geometric mismatch.\n');
end