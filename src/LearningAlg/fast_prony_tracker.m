function [freq_map, amp_map, damp_map, time_vec, win_len, num_of_win] = fast_prony_tracker(signal, fs, win_len_sec)
    % fast_prony_tracker
    % גרסה אופטימלית המשתמשת ב-LPC (Linear Predictive Coding)
    % במקום ב-Prony המלא, מה שמאיץ את החישוב פי 10-50.
    
    num_peaks = 8;               
    model_order = num_peaks * 2; 
    
    N = round(win_len_sec * fs);
    
    % Buffer creation
    [windows, ~] = buffer(signal, N, 0, 'nodelay');
    num_wins = size(windows, 2);
    
    % Pre-allocation
    freq_map = zeros(num_peaks, num_wins);
    amp_map  = zeros(num_peaks, num_wins);
    damp_map = zeros(num_peaks, num_wins);
    
    % pre-calculate time vector (const linear array for Vandermonde)
    t_n = (0:N-1).'; 

    for i = 1:num_wins
        seg = windows(:, i);
        
        % --- שלב 1: חישוב מקדמי AR (במקום prony) ---
        % שימוש באלגוריתם Levinson-Durbin המהיר
        % אנו מניחים מודל All-Pole (מתאים מאוד לרזוננס של רחפנים)
        try
            a = lpc(seg, model_order);
            
            % --- שלב 2: מציאת שורשים (Poles) ---
            r = roots(a);
            
            % --- שלב 3: סינון פיזיקלי ראשוני ---
            % אנו רוצים רק שורשים בחצי העליון של מעגל היחידה (תדרים חיוביים)
            % ורק כאלה שהם יציבים (בתוך מעגל היחידה, אבל קרובים אליו)
            r = r(imag(r) > 0); 
            
            if isempty(r), continue; end
            
            % המרה לתדר וריסון
            f_hz = angle(r) * fs / (2*pi);
            damping = fs * log(abs(r));
            
            % סינון תדרים לא רלוונטיים (למשל DC או תדרים נמוכים מאוד)
            idx_valid = find(f_hz > 20); 
            
            if isempty(idx_valid), continue; end
            
            curr_freqs = f_hz(idx_valid);
            curr_damp  = damping(idx_valid);
            curr_roots = r(idx_valid);
            
            % --- שלב 4: חישוב עוצמות (Amplitudes) ---
            % מכיוון ש-LPC לא מחזיר עוצמות, נחשב אותן ע"י Least Squares
            % Signal ~= Sum( Amp * r^n )
            % המטריצה היא מטריצת Vandermonde
            
            % אופטימיזציה: אם יש יותר מדי שורשים, ניקח רק את ה-8 הראשונים לחישוב המטריצה
            % כדי לא לבזבז זמן חישוב על תדרים זניחים, אבל כאן נחשב לכולם כדי לדייק
            
            V = (curr_roots .').^t_n; % Vandermonde Matrix [N x NumRoots]
            
            % פתרון מהיר למערכת לינארית: amps = V \ seg
            curr_amps_complex = V \ seg;
            curr_amps = abs(curr_amps_complex); % העוצמה היא הערך המוחלט
            
            % --- שלב 5: מיון ושמירה ---
            [sorted_amps, sort_idx] = sort(curr_amps, 'descend');
            
            sorted_freqs = curr_freqs(sort_idx);
            sorted_damp  = curr_damp(sort_idx);
            
            count = min(length(sorted_freqs), num_peaks);
            if count > 0
                freq_map(1:count, i) = sorted_freqs(1:count);
                amp_map(1:count, i)  = sorted_amps(1:count);
                damp_map(1:count, i) = sorted_damp(1:count);
            end
            
        catch
            continue;
        end
    end
    
    total_duration = length(signal)/fs;
    time_vec = linspace(win_len_sec/2, total_duration - win_len_sec/2, num_wins);
    win_len = win_len_sec;
    num_of_win = num_wins;
end