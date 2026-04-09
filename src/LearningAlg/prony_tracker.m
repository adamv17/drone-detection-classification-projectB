function [freq_map, amp_map, damp_map, time_vec, win_len, num_of_win] = prony_tracker(signal, fs, win_len_sec, order)
    % prony_tracker
    % מבצע פרוני ומחזיר תדרים, עוצמות וריסון (Damping).
    %
    % Outputs:
    %   freq_map - מטריצה [8 x NumWindows] (הרץ)
    %   amp_map  - מטריצה [8 x NumWindows] (עוצמה)
    %   damp_map - מטריצה [8 x NumWindows] (פקטור דעיכה - 1/sec)
    
    num_peaks = order;               
    model_order = num_peaks * 2; 
    
    N = round(win_len_sec * fs);
    [windows, ~] = buffer(signal, N, 0, 'nodelay');
    num_wins = size(windows, 2);
    
    % --- עדכון 1: הכנת מטריצה ל-Damping ---
    freq_map = zeros(num_peaks, num_wins);
    amp_map  = zeros(num_peaks, num_wins);
    damp_map = zeros(num_peaks, num_wins); % <--- חדש
    
    for i = 1:num_wins
        seg = windows(:, i);
        
        try
            [b, a] = prony(seg, model_order, model_order);
            [r, p, ~] = residuez(b, a);
            
            % המרה לפיזיקה
            f_hz = angle(p) * fs / (2*pi);
            mag  = abs(r);
            
            % --- עדכון 2: חישוב ה-Damping ---
            % הקשר הוא: p = exp(damping/fs + j*omega)
            % לכן: damping = fs * ln(|p|)
            damping = fs * log(abs(p)); 
            
            % סינון (רק תדרים חיוביים והגיוניים)
            idx_valid = find(f_hz > 10); 
            
            curr_freqs = f_hz(idx_valid);
            curr_amps  = mag(idx_valid);
            curr_damp  = damping(idx_valid); % <--- לוקחים את הריסון המתאים
            
            % מיון לפי עוצמה (כדי שהכי חזק יהיה ראשון)
            [sorted_amps, sort_idx] = sort(curr_amps, 'descend');
            
            sorted_freqs = curr_freqs(sort_idx);
            sorted_damp  = curr_damp(sort_idx);  % <--- ממיינים גם את הריסון לפי העוצמה
            
            % מילוי התוצאות
            count = min(length(sorted_freqs), num_peaks);
            if count > 0
                freq_map(1:count, i) = sorted_freqs(1:count);
                amp_map(1:count, i)  = sorted_amps(1:count);
                damp_map(1:count, i) = sorted_damp(1:count); % <--- שמירה במטריצה
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