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