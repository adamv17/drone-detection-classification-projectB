clear; clc; close all;

dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio';
wavFiles = dir(fullfile(dataPath, '*.wav'));
filesToProcess = [1, 5, 40, 56, 61, 66];

for i = filesToProcess % 1:length(wavFiles)
    baseFileName = wavFiles(i).name;
    fullFileName = fullfile(dataPath, baseFileName);
    nameParts = split(baseFileName, '_');
    trueLabel = nameParts{1};

    [signal, Fs] = audioread(fullFileName);
    if (size(signal,2) > 1)
        signal = signal(:, 1); % only take first channel if stereo
    end
    [classification, bfs, med_bfs, var_bfs] = droneDetection(signal, Fs);
     
    figure;
    t_sig = (0:length(signal)-1) / Fs;
    
    % Calculate time vector for the classification output
    % ---------------------------------------------------
    % Must match values in droneDetection function
    step_durr = 0.2; 
    var_limit_for_plot = 25;
    % ---------------------------------------------------

    t_class = (0:length(classification)-1) * step_durr;

    spec_win_dur = 0.1; % 100ms window
    spec_win_smp = min(length(signal), floor(spec_win_dur * Fs));
    spec_overlap_smp = floor(spec_win_smp / 2); % 50% overlap
    spec_nfft = 2^nextpow2(spec_win_smp); % FFT points
    
    subplot(3,1,1);
    spectrogram(signal, spec_win_smp, spec_overlap_smp, spec_nfft, Fs);
    title(['Spectrogram: ' baseFileName], 'Interpreter', 'none');
    xlabel('Time (s)');
    ylabel('Frequency (kHz)');
    
    % --- Plot 2: Frequency vs. Variance (Clearer Visualization) ---
    subplot(3,1,2);
    
    colororder({'b', 'r'}); % Set color order: blue, then red
    
    % Plot Frequency on the LEFT Y-AXIS
    yyaxis left;
    plot(t_class, med_bfs, 'b-', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Median Base Freq (Hz)');
    ylim([-10 max(med_bfs)+10]); 
    
    % Plot Variance on the RIGHT Y-AXIS
    yyaxis right;
    plot(t_class, var_bfs, 'r-', 'LineWidth', 1.5);
    hold on;
    % Plot the variance threshold line
    plot(t_class, ones(size(t_class)) * var_limit_for_plot, 'r--', 'LineWidth', 1);
    ylabel('Local Variance (Hz^2)');
    ylim([-5 max(var_bfs)]); % Adjust as needed to see variance clearly
    
    title(['Detection Analysis (True Label: ' trueLabel ')']);
    legend('Median BFS', 'Local Variance', 'Variance Threshold');
    grid on;

    subplot(3,1,3);
    plot(t_class, classification, 'r', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Classification')
    title(['Classification (True Label:' trueLabel ')']);
    ylim([-0.1 1.1]);
    grid on;
    hold off;

    fprintf('  (Plotting results... Close figure to continue)\n');
    waitfor(gcf);   
end

