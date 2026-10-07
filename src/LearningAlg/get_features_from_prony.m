function [std_vec, damp_matrix, freq_matrix  ] = get_features_from_prony(freq_map, amp_map ,damp_map, t_vec, win_len, num_of_win)

    [sorted_freqs, sort_idx] = sort(freq_map, 1, 'ascend');
    

    sorted_amps = zeros(size(amp_map));
    sorted_damp = zeros(size(damp_map));
    for col = 1:num_of_win
        % סידור העוצמות באותה עמודה לפי האינדקסים של מיון התדרים
        sorted_amps(:, col) = amp_map(sort_idx(:, col), col);
        sorted_damp(:, col) = damp_map(sort_idx(:, col), col);
    end
    
    % --- שלב 3: חישוב הוואריאנס לכל שורה ---

    std_vec = std(sorted_freqs, 0, 2, 'omitnan');

    % Return the frequency-sorted matrices so each row is a consistent mode
    % across windows; the raw maps are in arbitrary per-window order.
    damp_matrix = sorted_damp;

    freq_matrix = sorted_freqs;
    
    % התוצאה: וקטור עמודה (למשל 8x1)
end
