%% run_monte_carlo_analysis.m
% ---------------------------------------------------------
% PURPOSE: 
% 1. Runs Monte Carlo Cross-Validation (20 random splits).
% 2. Uses STRATIFIED splitting to ensure Drones exist in Test Set.
% 3. Aggregates Accuracy, Precision, Recall, F1, AUC, and FPR.
% 4. Generates Spectrograms + Dashboard.
% 5. SAVES all results to 'logs'.
% ---------------------------------------------------------

%% 0. LOAD AND MERGE DATA
fprintf('Loading datasets...\n');

% Load standard data
stdData = load('TrueFullTable.mat'); 
Table1 = stdData.FullTable;

% Load early-only data 
revData = load('RevTrueFullTable.mat');
Table2 = revData.FullTable;

% Vertically concatenate the two tables
FullTable = [Table1; Table2];

% Sort by Filename so the early frames and standard frames for 
% the same audio file are grouped together sequentially
FullTable = sortrows(FullTable, 'Filename');

fprintf('Successfully merged! Total combined frames: %d\n', height(FullTable));

dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio';

%% 1. CONFIGURATION & LOGGING SETUP
numIterations = 1; 
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
models_store = cell(numIterations, 1);

% Global Tracking
global_best.acc = -1; global_best.filename = "";
global_worst.acc = 101; global_worst.filename = "";

% --- FILE-LEVEL EVENT-DETECTION SWEEP (consumed by Section 4) ---
% A recording counts as a DRONE detection if its SMOOTHED probability crosses
% the threshold even once (peak > threshold). We sweep these thresholds so the
% operating point can be chosen; non-drone files that cross it are false alarms.
thresholds  = 0.05:0.05:0.95;          % swept decision thresholds
nThr        = numel(thresholds);
filePeaks   = cell(numIterations, 1);  % peak smoothed drone prob per test file
fileIsDrone = cell(numIterations, 1);  % ground-truth drone flag per test file

% Features
nonFeatureCols = {'Label', 'Filename'};
featureNames = setdiff(FullTable.Properties.VariableNames, nonFeatureCols, 'stable');

% --- STRATIFICATION PREP (OPTIMIZED) ---
fprintf('Mapping files to labels for stratification...\n');

% The 'idx' output gives the first row index where each unique file appears
[uniqueFiles, idx] = unique(FullTable.Filename, 'stable'); 

% Instantly extract the labels using those indices
uniqueLabels = cellstr(string(FullTable.Label(idx)));

fprintf('Starting Monte Carlo Validation (%d Iterations)...\n', numIterations);
numWorkers = 6;
delete(parcluster('Processes').Jobs)
parpool('Processes', numWorkers)

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
    opts = statset('UseParallel', true);
    rf = TreeBagger(numTrees, X_Tr, Y_Tr, 'Method', 'classification', ...
        'PredictorNames', featureNames, 'OOBPrediction', 'off', 'Options', opts);

    models_store{k} = compact(rf); % Store the model for this iteration
    
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

    % Collect per-file detection data for the Section 4 threshold sweep
    peakVec  = zeros(length(iterFiles), 1);
    droneVec = false(length(iterFiles), 1);

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
        % Ground truth from the LABEL column (authoritative), not the filename:
        % a file is a drone file if it contains any DRONE-labelled frame.
        isDroneFile = any(strcmpi(file_truth, 'DRONE'));

        % Store peak smoothed probability + truth for the threshold sweep
        peakVec(f)  = max(file_smooth);
        droneVec(f) = isDroneFile;

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

    % Persist this iteration's per-file detection data for the sweep
    filePeaks{k}   = peakVec;
    fileIsDrone{k} = droneVec;
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
models_store = models_store(valid_idx); % Filter models array

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

save(fullfile(logDir, 'results.mat'), 'acc_hist', 'prec_hist', 'rec_hist', 'f1_hist', 'fpr_hist', 'auc_hist', 'roc_store', 'global_best', 'global_worst','models_store', '-v7.3');

%% 4. FILE-LEVEL EVENT-DETECTION SWEEP (MODULAR ADD-ON)
% ---------------------------------------------------------
% Scores each recording by EVENT DETECTION instead of per-frame:
%   - a DRONE file is "correct" if the smoothed probability crosses the
%     threshold at least once (peak > threshold);
%   - a non-drone file that crosses it is a false alarm (symmetric rule).
% Sweeps every threshold, prints the table, and auto-selects the one with the
% best mean F1. Fully self-contained: consumes filePeaks / fileIsDrone only.
% ---------------------------------------------------------
fprintf('\nRunning file-level detection sweep...\n');

validK      = find(~cellfun(@isempty, filePeaks));
nValidSweep = numel(validK);

accSweep  = nan(nValidSweep, nThr);
precSweep = nan(nValidSweep, nThr);
recSweep  = nan(nValidSweep, nThr);
f1Sweep   = nan(nValidSweep, nThr);
fprSweep  = nan(nValidSweep, nThr);

for ii = 1:nValidSweep
    peakVec  = filePeaks{validK(ii)};
    droneVec = fileIsDrone{validK(ii)};
    for j = 1:nThr
        detected = peakVec > thresholds(j);      % "flagged even once" == peak > thr
        TP = sum( droneVec &  detected);
        FN = sum( droneVec & ~detected);
        FP = sum(~droneVec &  detected);
        TN = sum(~droneVec & ~detected);

        accSweep(ii, j)  = (TP + TN) / max(TP + TN + FP + FN, 1);
        precSweep(ii, j) = TP / (TP + FP);
        recSweep(ii, j)  = TP / (TP + FN);
        f1Sweep(ii, j)   = 2 * precSweep(ii, j) * recSweep(ii, j) / ...
                           (precSweep(ii, j) + recSweep(ii, j));
        fprSweep(ii, j)  = FP / (FP + TN);
    end
end
% Empty-class divisions -> 0
precSweep(isnan(precSweep)) = 0;
recSweep(isnan(recSweep))   = 0;
f1Sweep(isnan(f1Sweep))     = 0;
fprSweep(isnan(fprSweep))   = 0;

meanAcc  = mean(accSweep,  1);
meanPrec = mean(precSweep, 1);
meanRec  = mean(recSweep,  1);
meanF1   = mean(f1Sweep,   1);
meanFpr  = mean(fprSweep,  1);

[~, bestThrIdx]   = max(meanF1);
decisionThreshold = thresholds(bestThrIdx);
fprintf('Selected operating threshold = %.2f (mean F1 = %.3f, mean Acc = %.1f%%)\n', ...
    decisionThreshold, meanF1(bestThrIdx), meanAcc(bestThrIdx)*100);

% --- Save sweep report ---
sweepFile = fullfile(logDir, 'detection_sweep.txt');
fid = fopen(sweepFile, 'w');
fprintf(fid, '=================================================\n');
fprintf(fid, 'FILE-LEVEL EVENT-DETECTION SWEEP (%d valid runs)\n', nValidSweep);
fprintf(fid, 'Rule: drone file detected if smoothed probability\n');
fprintf(fid, 'crosses the threshold at least once (peak > threshold).\n');
fprintf(fid, 'Date: %s\n', timestamp);
fprintf(fid, '=================================================\n');
fprintf(fid, 'Selected operating threshold (max mean F1): %.2f\n', decisionThreshold);
fprintf(fid, '-------------------------------------------------\n');
fprintf(fid, '   Thr    Acc%%   Prec%%  Rec%%   F1%%    FPR%%\n');
for j = 1:nThr
    marker = ' ';
    if j == bestThrIdx, marker = '*'; end
    fprintf(fid, ' %s %.2f  %6.1f %6.1f %6.1f %6.1f %6.1f\n', marker, thresholds(j), ...
        meanAcc(j)*100, meanPrec(j)*100, meanRec(j)*100, meanF1(j)*100, meanFpr(j)*100);
end
fprintf(fid, '(* = selected operating threshold)\n');
fclose(fid);

save(fullfile(logDir, 'detection_sweep.mat'), 'thresholds', 'decisionThreshold', ...
    'accSweep', 'precSweep', 'recSweep', 'f1Sweep', 'fprSweep', ...
    'meanAcc', 'meanPrec', 'meanRec', 'meanF1', 'meanFpr', 'bestThrIdx', '-v7.3');

% --- Sweep figure: metrics-vs-threshold + confusion at operating point ---
fig_sweep = figure('Name', 'Detection Threshold Sweep', 'Color', 'w', ...
    'Position', [50, 50, 1200, 500], 'Visible', 'off');
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'File-Level Event-Detection Sweep', 'FontSize', 14, 'FontWeight', 'bold');

nexttile; hold on;
plot(thresholds, meanAcc,  '-o', 'LineWidth', 1.5, 'DisplayName', 'Accuracy');
plot(thresholds, meanPrec, '-s', 'LineWidth', 1.5, 'DisplayName', 'Precision');
plot(thresholds, meanRec,  '-^', 'LineWidth', 1.5, 'DisplayName', 'Recall');
plot(thresholds, meanF1,   '-d', 'LineWidth', 2.0, 'DisplayName', 'F1');
plot(thresholds, meanFpr,  '-x', 'LineWidth', 1.5, 'DisplayName', 'FPR');
xline(decisionThreshold, '--k', sprintf('Op. Thr = %.2f', decisionThreshold), ...
    'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
title('Metrics vs Threshold'); xlabel('Decision Threshold'); ylabel('Score');
ylim([0 1.05]); legend('Location', 'best'); grid on;

% File-level confusion at the operating threshold (last valid run)
peaksF = filePeaks{validK(end)};
isDrF  = fileIsDrone{validK(end)};
detF   = peaksF > decisionThreshold;
truthLab = categorical(isDrF, [true false], {'DRONE', 'OTHER'});
predLab  = categorical(detF,  [true false], {'DRONE', 'OTHER'});
nexttile;
confusionchart(truthLab, predLab, 'Title', sprintf('File Detection @ Thr=%.2f', decisionThreshold));

saveas(fig_sweep, fullfile(logDir, 'Detection_Sweep.png'));
close(fig_sweep);

fprintf('Detection sweep saved to logs.\n');

%% 5. EXPORT DEPLOYABLE MODEL BUNDLE
% ---------------------------------------------------------
% Saves ONE small, self-describing file you can reload on its own (no need to
% open the big results.mat). It bundles the model WITH everything required to
% use it: feature order, class names, the chosen threshold, and smoothing.
% Picks the iteration with the best file-level F1 at the operating threshold.
% ---------------------------------------------------------
[~, bestRun] = max(f1Sweep(:, bestThrIdx));

Model = struct();
Model.rf                = models_store{bestRun};        % CompactTreeBagger
Model.featureNames      = featureNames;                 % REQUIRED column order
Model.classNames        = models_store{bestRun}.ClassNames;
Model.decisionThreshold = decisionThreshold;            % file-level, from sweep
Model.smoothWin         = smoothWin;                    % movmean window
Model.trainedDate       = timestamp;
Model.iteration         = bestRun;

modelFile = fullfile(logDir, 'DroneModel.mat');
save(modelFile, 'Model');   % default (v7) format: small, fast, portable
fprintf('Deployable model bundle saved to: %s\n', modelFile);
%% 6. VISUALIZATION 1: BEST CASE
fig_best = plot_case_with_spectrogram(global_best, 'Best Case (Success)', dataPath);
saveas(fig_best, fullfile(logDir, 'Best_Case_Spectrogram.png'));
close(fig_best);

%% 7. VISUALIZATION 2: WORST CASE
fig_worst = plot_case_with_spectrogram(global_worst, 'Worst Case (Failure)', dataPath);
saveas(fig_worst, fullfile(logDir, 'Worst_Case_Spectrogram.png'));
close(fig_worst);

%% 8. VISUALIZATION 3: DASHBOARD
fig_metrics = figure('Name', 'Stability Dashboard', 'Color', 'w', 'Position', [50, 50, 1500, 500], 'Visible', 'off');
t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, sprintf('Model Stability Analysis (%d Iterations)', numValid), 'FontSize', 14, 'FontWeight', 'bold');

nexttile;
confusionchart(Y_Te, preds, 'Title', 'Confusion Matrix (Final Run)');

nexttile;
% ADDED FPR TO BOXPLOT
data_to_plot = [acc_hist, prec_hist, rec_hist, f1_hist, fpr_hist];
labels = {'Acc', 'Prec', 'Rec', 'F1', 'FPR'};
if size(data_to_plot, 1) > 1
    % Normal Monte Carlo behavior
    boxplot(data_to_plot, 'Labels', labels);
    title('Monte Carlo Validation Results');
else
    % Fallback for Single Iteration (Use a Bar chart instead)
    bar(data_to_plot);
    set(gca, 'XTickLabel', labels);
    title('Single Iteration Validation Results');
    ylabel('Score');
end

nexttile; hold on;
for k = 1:numValid
    if ~isempty(roc_store{k,1})
        plot(roc_store{k,1}, roc_store{k,2}, 'Color', [0.2 0.2 0.8 0.4], 'LineWidth', 1);
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
