%% run_monte_carlo_analysis.m
% ---------------------------------------------------------
% PURPOSE: 
% 1. Runs Monte Carlo Cross-Validation (20 random splits).
% 2. Uses STRATIFIED splitting to ensure Drones exist in Test Set.
% 3. Aggregates Accuracy, Precision, Recall, F1, AUC, and FPR.
% 4. Generates Spectrograms + Dashboard.
% 5. SAVES all results to 'logs'.
% ---------------------------------------------------------

if ~exist('FullTable', 'var')
    error('FullTable not found! Run "learner_setup.m" first.');
end

dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 

%% 1. CONFIGURATION & LOGGING SETUP
numIterations = 20; 
numTrees = 50;
smoothWin = 20;

% Setup Log Folder
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
logDir = fullfile('logs', sprintf('Run_%s', timestamp));
if ~exist(logDir, 'dir'), mkdir(logDir); end
fprintf('Results will be saved to: %s\n', logDir);

% Storage
acc_hist = zeros(numIterations, 1);
prec_hist = zeros(numIterations, 1);
rec_hist = zeros(numIterations, 1);
f1_hist = zeros(numIterations, 1);
auc_hist = zeros(numIterations, 1);
fpr_hist = zeros(numIterations, 1); % NEW: False Positive Rate
roc_store = cell(numIterations, 2); 

% Global Tracking
global_best.acc = -1; global_best.filename = "";
global_worst.acc = 101; global_worst.filename = "";

% Features
nonFeatureCols = {'Label', 'Filename'};
featureNames = setdiff(FullTable.Properties.VariableNames, nonFeatureCols, 'stable');

% --- STRATIFICATION PREP ---
uniqueFiles = unique(FullTable.Filename);
uniqueLabels = cell(length(uniqueFiles), 1);

fprintf('Mapping files to labels for stratification...\n');
for i = 1:length(uniqueFiles)
    idx = find(FullTable.Filename == uniqueFiles(i), 1);
    uniqueLabels{i} = char(FullTable.Label(idx));
end

fprintf('Starting Monte Carlo Validation (%d Iterations)...\n', numIterations);

%% 2. MONTE CARLO LOOP
for k = 1:numIterations
    fprintf('Iteration %d/%d... ', k, numIterations);
    
    % --- STEP A: STRATIFIED PARTITION BY FILE ---
    cv_files = cvpartition(uniqueLabels, 'HoldOut', 0.2);
    
    testFiles = uniqueFiles(test(cv_files));
    isTestRow = ismember(FullTable.Filename, testFiles);
    
    TrainT = FullTable(~isTestRow, :);
    TestT  = FullTable(isTestRow, :);
    
    X_Tr = TrainT{:, featureNames}; Y_Tr = TrainT.Label;
    X_Te = TestT{:, featureNames};  Y_Te = TestT.Label;
    
    % Safety check
    if length(unique(Y_Te)) < 2
        warning('Skipping iteration %d: Test set has only one class.', k);
        continue;
    end
    
    % --- STEP B: TRAIN ---
    rf = TreeBagger(numTrees, X_Tr, Y_Tr, 'Method', 'classification', ...
        'PredictorNames', featureNames, 'OOBPrediction', 'off');
    
    % --- STEP C: PREDICT ---
    [preds, scores] = predict(rf, X_Te);
    preds = categorical(preds);
    
    droneCol = find(strcmp(rf.ClassNames, 'DRONE'));
    raw_scores = scores(:, droneCol);
    
    % --- STEP D: METRICS ---
    C = confusionmat(Y_Te, preds);
    cls = categories(Y_Te);
    dIdx = find(strcmpi(cls, 'DRONE'));
    
    if size(C,1) < 2
        TP=0; FN=0; FP=0; TN=0; 
    else
        TP = C(dIdx, dIdx);
        FN = sum(C(dIdx, :)) - TP;
        FP = sum(C(:, dIdx)) - TP;
        TN = sum(C(:)) - (TP + FP + FN);
    end
    
    acc_hist(k) = (TP+TN) / sum(C(:));
    prec_hist(k) = TP / (TP+FP);
    rec_hist(k)  = TP / (TP+FN);
    f1_hist(k)   = 2*(prec_hist(k)*rec_hist(k)) / (prec_hist(k)+rec_hist(k));
    fpr_hist(k)  = FP / (FP + TN); % NEW: FPR Calculation
    
    if isnan(prec_hist(k)), prec_hist(k)=0; end
    if isnan(f1_hist(k)), f1_hist(k)=0; end
    if isnan(fpr_hist(k)), fpr_hist(k)=0; end
    
    [fpr, tpr, ~, auc_hist(k)] = perfcurve(Y_Te, raw_scores, 'DRONE');
    roc_store{k, 1} = fpr;
    roc_store{k, 2} = tpr;
    
    fprintf('Acc: %.2f%%, FPR: %.2f%%\n', acc_hist(k)*100, fpr_hist(k)*100);
    
    % --- STEP E: FIND BEST/WORST FILES ---
    testFileNames = TestT.Filename;
    iterFiles = unique(testFileNames);
    
    for f = 1:length(iterFiles)
        fname = iterFiles(f);
        idx = strcmp(testFileNames, fname);
        
        file_truth = Y_Te(idx);
        file_raw   = raw_scores(idx);
        
        % Temporal Smoothing
        file_smooth = movmean(file_raw, smoothWin); 
        
        % Decision based on Smoothed Score
        file_decisions = categorical(repmat({'OTHER'}, length(file_smooth), 1));
        file_decisions(file_smooth > 0.5) = 'DRONE';
        
        if ~iscell(file_decisions), file_decisions = cellstr(file_decisions); end
        if ~iscell(file_truth), file_truth = cellstr(file_truth); end
        
        file_acc = mean(strcmp(file_truth, file_decisions));
        isDroneFile = contains(string(fname), 'DRONE', 'IgnoreCase', true);
        
        if isDroneFile && (file_acc > global_best.acc)
            global_best.acc = file_acc;
            global_best.filename = fname;
            global_best.scores_raw = file_raw;
            global_best.scores_smooth = file_smooth;
            global_best.truth = strcmpi(file_truth, 'DRONE');
        end
        if isDroneFile && (file_acc < global_worst.acc)
            global_worst.acc = file_acc;
            global_worst.filename = fname;
            global_worst.scores_raw = file_raw;
            global_worst.scores_smooth = file_smooth;
            global_worst.truth = strcmpi(file_truth, 'DRONE');
        end
    end
end

% Filter valid runs
valid_idx = acc_hist > 0;
acc_hist = acc_hist(valid_idx);
prec_hist = prec_hist(valid_idx);
rec_hist = rec_hist(valid_idx);
f1_hist = f1_hist(valid_idx);
auc_hist = auc_hist(valid_idx);
fpr_hist = fpr_hist(valid_idx); % Filter FPR
roc_store = roc_store(valid_idx, :);

numValid = sum(valid_idx);

%% 3. REPORT GENERATION & SAVING
fprintf('\nSaving Results to %s ...\n', logDir);

summaryFile = fullfile(logDir, 'metrics_summary.txt');
fid = fopen(summaryFile, 'w');
fprintf(fid, '=================================================\n');
fprintf(fid, 'INTERMEDIATE RESULTS (Aggregated over %d Valid Runs)\n', numValid);
fprintf(fid, 'Date: %s\n', timestamp);
fprintf(fid, '=================================================\n');
fprintf(fid, 'Accuracy:  %.2f%% (+/- %.2f)\n', mean(acc_hist)*100, std(acc_hist)*100);
fprintf(fid, 'Precision: %.2f%% (+/- %.2f)\n', mean(prec_hist)*100, std(prec_hist)*100);
fprintf(fid, 'Recall:    %.2f%% (+/- %.2f)\n', mean(rec_hist)*100, std(rec_hist)*100);
fprintf(fid, 'F1-Score:  %.2f%% (+/- %.2f)\n', mean(f1_hist)*100, std(f1_hist)*100);
fprintf(fid, 'FPR:       %.2f%% (+/- %.2f)\n', mean(fpr_hist)*100, std(fpr_hist)*100); % NEW
fprintf(fid, 'AUC:       %.3f  (+/- %.3f)\n', mean(auc_hist), std(auc_hist));
fprintf(fid, '-------------------------------------------------\n');
fprintf(fid, 'Best Case File: %s (Acc: %.2f%%)\n', global_best.filename, global_best.acc*100);
fprintf(fid, 'Worst Case File: %s (Acc: %.2f%%)\n', global_worst.filename, global_worst.acc*100);
fclose(fid);

save(fullfile(logDir, 'results.mat'), 'acc_hist', 'prec_hist', 'rec_hist', 'f1_hist', 'fpr_hist', 'auc_hist', 'roc_store', 'global_best', 'global_worst');

%% 4. VISUALIZATION 1: BEST CASE
fig_best = plot_case_with_spectrogram(global_best, 'Best Case (Success)', dataPath);
saveas(fig_best, fullfile(logDir, 'Best_Case_Spectrogram.png'));
close(fig_best);

%% 5. VISUALIZATION 2: WORST CASE
fig_worst = plot_case_with_spectrogram(global_worst, 'Worst Case (Failure)', dataPath);
saveas(fig_worst, fullfile(logDir, 'Worst_Case_Spectrogram.png'));
close(fig_worst);

%% 6. VISUALIZATION 3: DASHBOARD
fig_metrics = figure('Name', 'Stability Dashboard', 'Color', 'w', 'Position', [50, 50, 1500, 500], 'Visible', 'off');
t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, sprintf('Model Stability Analysis (%d Iterations)', numValid), 'FontSize', 14, 'FontWeight', 'bold');

nexttile;
confusionchart(Y_Te, preds, 'Title', 'Confusion Matrix (Final Run)');

nexttile;
% ADDED FPR TO BOXPLOT
boxplot([acc_hist, prec_hist, rec_hist, f1_hist, fpr_hist], 'Labels', {'Acc', 'Prec', 'Rec', 'F1', 'FPR'});
title('Metric Variance'); ylabel('Score (0-1)'); grid on;
ylim([-0.05 1.05]); 

nexttile; hold on;
for k = 1:numValid
    if ~isempty(roc_store{k,1})
        plot(roc_store{k,1}, roc_store{k,2}, 'Color', [0.7 0.7 0.7 0.4], 'LineWidth', 1);
    end
end
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5);
title(sprintf('ROC Curves (Mean AUC: %.3f)', mean(auc_hist)));
xlabel('False Positive Rate'); ylabel('True Positive Rate');
grid on; axis square;

saveas(fig_metrics, fullfile(logDir, 'Metrics_Dashboard.png'));
close(fig_metrics);

fprintf('All figures saved to logs.\n');

%% --- LOCAL FUNCTIONS ---
function fig = plot_case_with_spectrogram(caseData, titleStr, audioPath)
    if caseData.filename == ""
        warning('No case data found for %s', titleStr);
        fig = figure; return;
    end
    
    full_wav_path = fullfile(audioPath, char(caseData.filename));
    [y, fs] = audioread(full_wav_path);
    if size(y, 2) > 1, y = y(:, 1); end 
    
    numFrames = length(caseData.scores_raw);
    t_prob = linspace(0, length(y)/fs, numFrames);
    
    fig = figure('Name', titleStr, 'Color', 'w', 'Position', [50, 50, 1200, 700], 'Visible', 'off');
    t = tiledlayout(2, 1, 'TileSpacing', 'tight');
    
    nexttile; hold on;
    area(t_prob, double(caseData.truth), 'FaceColor', [0.9 0.95 0.9], 'EdgeColor', 'none', 'DisplayName', 'Ground Truth');
    plot(t_prob, caseData.scores_raw, 'Color', [0.7 0.7 0.7], 'LineStyle', ':', 'LineWidth', 1, 'DisplayName', 'Raw RF Score');
    plot(t_prob, caseData.scores_smooth, 'Color', '#0072BD', 'LineWidth', 2, 'DisplayName', 'Smoothed Probability');
    yline(0.5, '--r', 'Decision Threshold');
    title(sprintf('%s: %s (Acc: %.1f%%)', titleStr, caseData.filename, caseData.acc*100), 'Interpreter', 'none', 'FontSize', 14);
    ylabel('Drone Probability'); legend('Location', 'best'); grid on; ylim([-0.1 1.1]); xlim([0 length(y)/fs]);
    
    nexttile;
    window = round(0.05*fs); noverlap = round(0.04*fs); nfft = 4096;
    [S, F, T_spec, P] = spectrogram(y, window, noverlap, nfft, fs, 'yaxis');
    
    imagesc(T_spec, F/1000, 10*log10(abs(P))); 
    axis xy; 
    colormap('jet');  
    colorbar; 
    caxis([-100 -20]); 
    
    title('Spectral Analysis (Spectrogram)'); ylabel('Frequency (kHz)'); xlabel('Time (s)');
    ylim([0 10]); xlim([0 length(y)/fs]);
    linkaxes(findobj(fig, 'Type', 'axes'), 'x');
end