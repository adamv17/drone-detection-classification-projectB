%% run_rf_analysis.m
% ---------------------------------------------------------
% PURPOSE: 
% 1. Takes 'TrainTable' and 'TestTable' from the workspace.
% 2. Trains a Random Forest (TreeBagger) with OOB Permutation.
% 3. Visualizes Feature Importance (Fixed Labels).
% 4. Visualizes a Single Decision Tree.
% 5. Validates on the Test Set.
% ---------------------------------------------------------

if ~exist('TrainTable', 'var') || ~exist('TestTable', 'var')
    error('Data not found! Please run "learner_setup.m" first.');
end

fprintf('Loading data from workspace...\n');
varNames = TrainTable.Properties.VariableNames(1:end-1);
X_Train = TrainTable{:, 1:end-1}; 
Y_Train = TrainTable.Label;

%% 1. TRAIN RANDOM FOREST
fprintf('Training Random Forest (50 Trees)...\n');

numTrees = 50;
rf_model = TreeBagger(numTrees, X_Train, Y_Train, ...
    'Method', 'classification', ...
    'OOBPredictorImportance', 'on', ... 
    'OOBPrediction', 'on', ...
    'PredictorNames', varNames); 

fprintf('Training Complete.\n');

%% 2. VISUALIZE FEATURE IMPORTANCE (FIXED LABELS)
importance_scores = rf_model.OOBPermutedPredictorDeltaError;
[sorted_scores, sorted_idx] = sort(importance_scores, 'descend');

figure('Position', [100, 100, 1200, 500], 'Color', 'w');

% Plot A: Global Overview
subplot(1, 2, 1);
bar(importance_scores);
xlabel('Feature Index'); 
ylabel('Importance (Delta Error)');
title('Global Feature Importance'); 
grid on;

% Plot B: Top 15 Winners
subplot(1, 2, 2);
topN = min(15, length(varNames)); 
barh(sorted_scores(1:topN));

% --- THE FIX FOR LABELS ---
% We explicitly set the TickLabelInterpreter to 'none'.
% This stops MATLAB from treating '_' as a subscript command.
set(gca, 'YTick', 1:topN, ...
    'YTickLabel', varNames(sorted_idx(1:topN)), ...
    'TickLabelInterpreter', 'none'); 
% --------------------------

xlabel('Importance Score'); 
title(['Top ' num2str(topN) ' Features for Drone Detection']);
grid on;

%% 3. TEST SET VALIDATION
fprintf('\nEvaluating on Test Set...\n');

X_Test = TestTable{:, 1:end-1};
Y_Test = TestTable.Label;

[pred_labels, scores] = predict(rf_model, X_Test);
pred_labels = categorical(pred_labels);

accuracy = sum(pred_labels == Y_Test) / length(Y_Test) * 100;
fprintf('Test Set Accuracy: %.2f%%\n', accuracy);

figure('Name', 'Confusion Matrix', 'Color', 'w');
cm = confusionchart(Y_Test, pred_labels);
cm.Title = ['Test Accuracy: ' num2str(accuracy, '%.2f') '%'];

%% 4. VISUALIZE A SINGLE TREE (NEW SECTION)
fprintf('\nVisualizing Tree #1...\n');

% Extract the first trained tree
first_tree = rf_model.Trees{1}; 

% Create a dedicated figure for the tree view
figure('Name', 'Decision Tree Visualization', 'Color', 'w');

% 'Mode', 'graph' creates the flowchart view
view(first_tree, 'Mode', 'graph'); 
title('Visualizing Tree #1 of the Forest');

% Optional: Print interpretation
disp('------------------------------------------------');
disp('TREE VISUALIZATION GUIDE:');
disp('1. Triangles are Decision Nodes (e.g., if mfcc_1 < 0.5 go left).');
disp('2. Circles are Leaf Nodes (Final Decision: Drone or Other).');
disp('3. Zoom in with the magnifying glass tool to see the split logic.');
disp('------------------------------------------------');