function feat = getBicoherenceFeature(x, fs)
    % 1. Constants
    nfft = 256; 
    overlap = 0.75; 
    
    % 2. Bicoherence Calculation
    window = hamming(nfft);
    noverlap = floor(nfft * overlap);
    
    X_seg = buffer(x, nfft, noverlap, 'nodelay');
    X_seg = X_seg .* window;
    X_fft = fft(X_seg, nfft);
    X_fft = X_fft(1:nfft/2+1, :); 
    [numFreqs, numSegs] = size(X_fft);
    
    % Safety return (Size must match the 11 features below)
    if numSegs < 5
        feat = zeros(1, 11); 
        return;
    end

    Bispectrum = zeros(numFreqs, numFreqs);
    P_spec = zeros(numFreqs, 1);
    
    for k = 1:numSegs
        Xk = X_fft(:, k); 
        XX = Xk * Xk.'; 
        [I, J] = meshgrid(1:numFreqs, 1:numFreqs);
        sum_idx = I + J - 1; 
        mask = sum_idx <= numFreqs;
        X_sum = zeros(numFreqs, numFreqs);
        X_sum(mask) = conj(Xk(sum_idx(mask)));
        Bispectrum = Bispectrum + (XX .* X_sum);
        P_spec = P_spec + abs(Xk).^2;
    end
    
    Bispectrum = Bispectrum / numSegs;
    P_spec = P_spec / numSegs;
    
    [P1, P2] = meshgrid(P_spec, P_spec);
    P_sum = zeros(numFreqs, numFreqs);
    P_sum(mask) = P_spec(sum_idx(mask));
    denom = (P1 .* P2) .* P_sum;
    
    % The Raw 2D Map
    bic2 = (abs(Bispectrum).^2) ./ (denom + 1e-12);
    
    % --- 3. APPLY STATISTICAL MASK --
    % 0.95 confidence of statistical significance
    significance_threshold = 3/numSegs; 
    
    % Clean Map (Noise forced to zero) for AIB/Sum
    bic2_clean = bic2;
    bic2_clean(bic2 < significance_threshold) = 0; 
    
    % 4. FEATURE EXTRACTION (NO HISTOGRAMS)
    
    freq_axis = linspace(0, fs/2, numFreqs);
    [F1, F2] = meshgrid(freq_axis, freq_axis);
    geom_mask = (F2 <= F1) & ((F1 + F2) <= fs/2);
    
    mask_low = geom_mask & (F1 < 2000);
    mask_high = geom_mask & (F1 > 10000);
    
    % Use Raw for Max (to see single peaks)
    pixels_all = bic2(geom_mask);
    pixels_low = bic2(mask_low);
    pixels_high = bic2(mask_high);
    
    % Use Clean for Sums (to avoid noise accumulation)
    clean_all = bic2_clean(geom_mask);
    clean_low = bic2_clean(mask_low);
    clean_high = bic2_clean(mask_high);

    % A. Max Peaks 
    max_all = max(pixels_all, [], 'all'); if isempty(max_all), max_all=0; end
    max_low = max(pixels_low, [], 'all'); if isempty(max_low), max_low=0; end
    max_high = max(pixels_high, [], 'all'); if isempty(max_high), max_high=0; end
    
    % B. Sum Significant Energy
    sum_sig_all = sum(clean_all);
    sum_sig_low = sum(clean_low);
    sum_sig_high = sum(clean_high);
    
    % C. Axial Integrated Bispectrum (AIB) - Uses CLEAN map
    raw_aib = sum(bic2_clean, 2); 
    
    idx_heli     = freq_axis < 100;              % Big Rotor
    idx_drone    = (freq_axis >= 100) & (freq_axis < 300); % Small Rotor
    idx_harmonic = (freq_axis >= 300) & (freq_axis < 1000);
    idx_mid      = (freq_axis >= 1000) & (freq_axis < 5000);
    idx_empty    = (freq_axis >= 5000) & (freq_axis < 10000);
    idx_pwm      = freq_axis >= 10000;
    
    aib_heli     = mean(raw_aib(idx_heli));
    aib_drone    = mean(raw_aib(idx_drone));
    aib_harmonic = mean(raw_aib(idx_harmonic));
    aib_mid      = mean(raw_aib(idx_mid));
    aib_empty    = mean(raw_aib(idx_empty));
    aib_pwm      = mean(raw_aib(idx_pwm));
    
    aib_feats = [aib_heli, aib_drone, aib_harmonic, aib_mid, aib_empty, aib_pwm];
    aib_feats(isnan(aib_feats)) = 0;

    feat = [%max_all, max_low, max_high, ...
            %sum_sig_all, sum_sig_low, sum_sig_high, ...
            aib_feats];       
    feat(isnan(feat)) = 0;
end
