function [variance_vec] = get_features_from_prony(freq_map, amp_map, t_vec, win_len, num_of_win)
    % get_features_from_prony
    % הפונקציה מחשבת את השונות (Variance) של התדרים לאורך הזמן,
    % לאחר שהיא ממיינת אותם מהנמוך לגבוה בכל חלון זמן.
    
    % --- שלב 1: מיון התדרים לפי גובה (Hz) ---
    % 1 = מיון לאורך העמודות
    % 'ascend' = מהנמוך לגבוה
    [sorted_freqs, sort_idx] = sort(freq_map, 1, 'ascend');
    
    % --- שלב 2: שמירה על הצימוד (Coupling) של האמפליטודות ---
    % שלב זה קריטי אם נרצה בעתיד לחשב פיצ'רים על העוצמות לפי סדר התדרים.
    % כרגע זה מתבצע "ברקע" למען הסדר הטוב.
    sorted_amps = zeros(size(amp_map));
    for col = 1:num_of_win
        % סידור העוצמות באותה עמודה לפי האינדקסים של מיון התדרים
        sorted_amps(:, col) = amp_map(sort_idx(:, col), col);
    end
    
    % --- שלב 3: חישוב הוואריאנס לכל שורה ---
    % כעת, שורה 1 מכילה את התדר הכי נמוך בכל החלונות,
    % שורה 2 את התדר השני הכי נמוך, וכו'.
    % אנו מחשבים כמה התדרים האלו "זזים" או משתנים לאורך כל ההקלטה.
    
    % 0 = נרמול סטנדרטי (N-1)
    % 2 = ביצוע החישוב לרוחב (לאורך השורות)
    variance_vec = std(sorted_freqs, 0, 2, 'omitnan');
    
    % התוצאה: וקטור עמודה (למשל 8x1)
end


folder = "C:\Users\yahal\OneDrive\מסמכים\GitHub\drone-detection-classification-projectB\datasets\Drone-detection-dataset-master\Data\Audio";   % כאן שים את הנתיב לתיקייה
files = dir(fullfile(folder, '*.wav'));
X = 20;
[audio, fs] = audioread(fullfile(folder, files(X).name));
audio = audio(:,1);

%% הרצת אלגוריתם פרוני והצגת תוצאות
% 1. הגדרת פרמטרים
win_len_sec = 0.030; % חלון של 30 מילישניות

% 2. קריאה לפונקציה שבנית
[freq_map, amp_map, t_vec, win_len, num_of_win] = prony_tracker(audio, fs, win_len_sec);
var_vec=get_features_from_prony(freq_map, amp_map, t_vec, win_len, num_of_win)
