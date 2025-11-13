% -----------------
% Drone Detection  
% -----------------

function [classification_sig, base_sig, median_base_sig, var_base_sig] = droneDetection(signal, fs)
    % --- 1. Define Detection Parameters (from Report Chapter 4) ---
    HPcutoff = 500;       % [Hz] High-pass filter cutoff (Chapter 4.1), also changed
    detect_durr = 0.5;    % [sec] Classification input window duration
    step_durr = 0.2;      % [sec] Step size between windows (for overlap)
                          % (Using 0.2s from integration chapter 5.1)
    
    % Legal frequency range for motor (Chapter 4)
    % This was changed for electric drones
    f_L = 500;           % [Hz]
    f_H = 1e4;           % [Hz]
    
    % Classifier parameters (Chapter 4.2)
    lookback_med_dur = 1.0; % [sec] Causal median filter duration (example)
    lookback_var_dur = 2.0; % [sec] Causal variance/mean filter duration (example)
    var_limit = 25;       % [Hz^2] Variance threshold (T)
    alpha = 100;           % Penalty factor for sigmoid (example)
    
    % --- 2. Process Signal in Windows ---
    detect_smp = floor(detect_durr * fs);
    step_smp = floor(step_durr * fs);
    
    if detect_smp <= 0 || step_smp <= 0
        error('Window or step duration is too short for the sample rate.');
    end
    
    % Get window start indices
    window_starts = 1:step_smp:(length(signal) - detect_smp + 1);
    N_of_wins = length(window_starts);
    
    if N_of_wins == 0
        warning('Signal is too short for the specified window duration.');
        classification_sig = []; base_sig = []; median_base_sig = []; var_base_sig = [];
        return;
    end
    
    base_sig = zeros(1, N_of_wins);
    
    % --- 3. Run Base Frequency Estimator (BFE) on each window ---
    for i = 1:N_of_wins
        window_start = window_starts(i);
        window_end = window_start + detect_smp - 1;
        window = signal(window_start:window_end);
        
        % Call BFE
        base_sig(i) = bfe(window, fs, HPcutoff);
    end
    
    % --- 4. Apply Causal Median Filter (Chapter 4.2) ---
    % Calculate window size in samples
    med_window_smp = max(1, floor(lookback_med_dur / step_durr));
    % 'movmedian' with [k-1, 0] window is causal (looks back k-1 samples)
    median_base_sig = movmedian(base_sig, [med_window_smp-1, 0]);
    
    % --- 5. Apply Causal Variance & Mean Filters (Chapter 4.2) ---
    var_window_smp = max(1, floor(lookback_var_dur / step_durr));
    % 'movvar' and 'movmean' with [k-1, 0] are causal
    var_base_sig = movvar(median_base_sig, [var_window_smp-1, 0]);
    mean_base_sig = movmean(median_base_sig, [var_window_smp-1, 0]);
    
    % --- 6. Classify (Chapter 4.2) ---
    classification_sig = zeros(1, N_of_wins);
    for i = 1:N_of_wins
        V = var_base_sig(i);
        E = mean_base_sig(i);
        T = var_limit;
    
        % Calculate penalty terms using sigmoid function
        % (alpha * sigma(E - f_H) + 1)
        penalty_high = (alpha * sigmoid(E - f_H) + 1);
        % (alpha * sigma(f_L - E) + 1)
        penalty_low = (alpha * sigmoid(f_L - E) + 1);
        
        % Combined metric (Chapter 4.2, classification formula)
        metric = V * penalty_high * penalty_low;
        
        if metric < T
            classification_sig(i) = 1;
        else
            classification_sig(i) = 0;
        end
    end
end