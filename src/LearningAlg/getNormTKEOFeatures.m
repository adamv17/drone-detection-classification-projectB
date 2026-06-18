function feat = getNormTKEOFeatures(x)
    % 1. Compute TKEO Signal
    % Formula: Psi[n] = x[n]^2 - x[n-1]*x[n+1]
    
    % Vectorized implementation (Fast)
    % We lose 1 sample at start and 1 at end
    x_curr = x(2:end-1);
    x_prev = x(1:end-2);
    x_next = x(3:end);
    
    noise_floor = 1e-6;
    tkeo_sig = (x_curr.^2) - (x_prev .* x_next);
    sig_power = max(var(x), noise_floor); % Add noise floor
    tkeo_norm = tkeo_sig / sig_power;
    
    % 2. Extract Statistics
    % These summarize the "impulsiveness" of the frame
    
    val_mean = mean(tkeo_norm);
    val_std  = std(tkeo_norm);
    val_max  = max(tkeo_norm);
    
    % Kurtosis measures "spikiness". 
    % High kurtosis = Periodic mechanical clicks. 
    % Low kurtosis = Random Gaussian noise (Wind).
    val_kurt = kurtosis(tkeo_norm);
    
    feat = [val_mean, val_std, val_max, val_kurt];
    
    % Handle NaNs (e.g., if signal is perfectly flat zero)
    feat(isnan(feat)) = 0;
end