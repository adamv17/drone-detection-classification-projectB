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


    mean_vec = mean(sorted_freqs, 2, 'omitnan');
    std_vec = std(sorted_freqs, 0, 2, 'omitnan');
    
    damp_matrix=sorted_damp;

    freq_matrix = sorted_freqs;
    
   
    num_modes = size(freq_map, 1);
    tracked_amps = zeros(num_modes, num_of_win);

    for col = 1:num_of_win
        current_freqs = freq_map(:, col);
        current_amps = amp_map(:, col);
        
        for i = 1:num_modes
            % חישוב המרחק של כל התדרים מהתוחלת של הרכיב ה-i
            distances = abs(current_freqs - mean_vec(i));
            
            % מציאת התדר שהכי קרוב לתוחלת (בתוך הטווח שבין mean-std ל-mean+std)
            [min_dist, best_idx] = min(distances);
            
            % שליפת האמפליטודה המתאימה
            tracked_amps(i, col) = current_amps(best_idx);
        end
    end
    
    t_windows = linspace(t_vec(1), t_vec(end), num_of_win);

    figure;
    hold on;
    % מעבר על כל שורה (כל רכיב תדר שעקבנו אחריו)
    for i = 1:size(tracked_amps, 1)
        plot(t_windows, tracked_amps(i, :), 'LineWidth', 1.5, ...
            'DisplayName', ['Mode ' num2str(i) ' (avg: ' num2str(mean_vec(i), '%.1f') ' Hz)']);
    end
    
    grid on;
    xlabel('Time [sec]');
    ylabel('Amplitude');
    title('Tracked Amplitudes over Time');
    legend('Location', 'northeastoutside'); % הצגת מקרא מחוץ לגרף
    hold off;

end

function [distinction_string] = distinct(file1,file2)
win_len_sec = 0.030;
folder = "C:\Users\yahal\OneDrive\מסמכים\GitHub\drone-detection-classification-projectB\datasets\Drone-detection-dataset-master\Data\Audio";   % כאן שים את הנתיב לתיקייה
files = dir(fullfile(folder, '*.wav'));
[audio1, fs1] = audioread(fullfile(folder, files(file1).name));
audio1 = audio1(:,1);
[audio2, fs2] = audioread(fullfile(folder, files(file2).name));
audio2 = audio2(:,1);
[freq_map, amp_map,damp_map, t_vec, win_len, num_of_win] = prony_tracker(audio, fs, win_len_sec);
[freq_map, amp_map,damp_map, t_vec, win_len, num_of_win] = prony_tracker(audio, fs, win_len_sec);

end
folder = "C:\Users\yahal\OneDrive\מסמכים\GitHub\drone-detection-classification-projectB\datasets\Drone-detection-dataset-master\Data\Audio";   % כאן שים את הנתיב לתיקייה
files = dir(fullfile(folder, '*.wav'));
X = 88;
[audio, fs] = audioread(fullfile(folder, files(X).name));
audio = audio(:,1);

%% הרצת אלגוריתם פרוני והצגת תוצאות
% 1. הגדרת פרמטרים
win_len_sec = 0.030; % חלון של 30 מילישניות

% 2. קריאה לפונקציה שבנית
[freq_map, amp_map,damp_map, t_vec, win_len, num_of_win] = prony_tracker(audio, fs, win_len_sec);
var_vec=get_features_from_prony(freq_map, amp_map,damp_map, t_vec, win_len, num_of_win)
