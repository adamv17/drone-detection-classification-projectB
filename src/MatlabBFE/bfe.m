function estimated_bfs = bfe(window, fs, HPcutoff)
    % Base Frequency Estimator (BFE) as described in Chapter 4.1.
    
    n_window = length(window);
    if n_window == 0
        estimated_bfs = 0;
        return;
    end

    nfft = 2^nextpow2(n_window);
    
    % --- Step 1: HPF ---
    Y = fft(window, nfft);
    P2 = abs(Y / n_window);
    P1 = P2(1 : nfft/2 + 1);
    P1(2:end-1) = 2 * P1(2:end-1);
    
    f = fs * (0:(nfft/2)) / nfft;
    P1_db = 20 * log10(P1);
    P1_db(isinf(P1_db) | isnan(P1_db)) = -Inf; % Handle -Inf/NaN

    % Apply HPF in frequency domain
    filtered_P1_db = P1_db;
    filtered_P1_db(f < HPcutoff) = -Inf; % Set power below cutoff to min

    % --- Step 2: Find MDF ---
    [~, mdf_idx] = max(filtered_P1_db);
    MDF_freq = f(mdf_idx);
    
    if isinf(MDF_freq) || isnan(MDF_freq) || MDF_freq == 0
        estimated_bfs = 0; % No dominant frequency found
        return;
    end

    % --- Step 3: Determine Harmonic Order ---
    num_hypotheses = 5;
    set_scores = -Inf(1, num_hypotheses);
    base_freq_hypotheses = zeros(1, num_hypotheses);

    for h = 1:num_hypotheses % h = hypothesized harmonic order of MDF
        current_base_freq = MDF_freq / h;
        base_freq_hypotheses(h) = current_base_freq;
        
        harmonics_set = current_base_freq * (1:num_hypotheses);
        
        set_powers = [];
        for k = 1:num_hypotheses
            freq_k = harmonics_set(k);
            
            % Only score harmonics above the HPF cutoff (Chapter 4.1)
            if freq_k > HPcutoff
                % Find power at the closest frequency bin
                [~, freq_idx] = min(abs(f - freq_k));
                set_powers(end+1) = filtered_P1_db(freq_idx);
            end
        end
        
        if ~isempty(set_powers)
            set_scores(h) = mean(set_powers);
        end
    end

    % --- Step 4: Estimate Base Frequency ---
    [~, best_h_idx] = max(set_scores);
    estimated_bfs = base_freq_hypotheses(best_h_idx);
end