function [freq_map, amp_map, time_vec] = prony_tracker(signal, fs, win_len_sec)
    % prony_tracker
    % מבצע פרוני על חלונות זמן ומחזיר את 8 התדרים החזקים בכל רגע.
    %
    % Inputs:
    %   signal      - וקטור האודיו
    %   fs          - תדר דגימה
    %   win_len_sec - אורך חלון בשניות (למשל 0.1)
    %
    % Outputs:
    %   freq_map    - מטריצה [8 x NumWindows] של התדרים (Hz)
    %   amp_map     - מטריצה [8 x NumWindows] של העוצמות
    %   time_vec    - ציר הזמן
    
    num_peaks = 8;               % כמה תדרים לשמור
    model_order = num_peaks * 2; % סדר המודל
    
    % הגדרת גודל חלון בדגימות
    N = round(win_len_sec * fs);
    
    % חלוקה לחלונות (buffer) - ללא חפיפה (0 overlap) כדי שיהיה מהיר
    [windows, ~] = buffer(signal, N, 0, 'nodelay');
    num_wins = size(windows, 2);
    
    % הכנת מטריצות לתוצאות
    freq_map = zeros(num_peaks, num_wins);
    amp_map  = zeros(num_peaks, num_wins);
    
    % לולאה על כל חלון
    for i = 1:num_wins
        seg = windows(:, i);
        
        % 1. חישוב מקדמי פרוני
        % (משתמשים ב-try-catch למקרה של חלון שקט לחלוטין/שגיאה סינגולרית)
        try
            [b, a] = prony(seg, model_order, model_order);
            
            % 2. חישוב Residues כדי להשיג אמפליטודות
            [r, p, ~] = residuez(b, a);
            
            % 3. המרה לתדרים ועוצמות
            f_hz = angle(p) * fs / (2*pi);
            mag  = abs(r);
            
            % 4. סינון: רק תדרים חיוביים והגיוניים (מעל 10 הרץ למשל כדי לסנן DC)
            idx_valid = find(f_hz > 10); 
            
            curr_freqs = f_hz(idx_valid);
            curr_amps  = mag(idx_valid);
            
            % 5. מיון לפי עוצמה (מהחזק לחלש)
            [sorted_amps, sort_idx] = sort(curr_amps, 'descend');
            sorted_freqs = curr_freqs(sort_idx);
            
            % 6. מילוי התוצאות (לוקחים עד 8, או פחות אם אין)
            count = min(length(sorted_freqs), num_peaks);
            if count > 0
                freq_map(1:count, i) = sorted_freqs(1:count);
                amp_map(1:count, i)  = sorted_amps(1:count);
            end
            
        catch
            % אם החישוב נכשל בחלון מסוים (למשל שקט מוחלט), נשאיר אפסים
            continue;
        end
    end
    
    % יצירת ציר זמן (אמצע כל חלון)
    total_duration = length(signal)/fs;
    time_vec = linspace(win_len_sec/2, total_duration - win_len_sec/2, num_wins);
end


%folder = "C:\Users\yahal\OneDrive\מסמכים\GitHub\drone-detection-classification-projectB\datasets\Drone-detection-dataset-master\Data\Audio";   % כאן שים את הנתיב לתיקייה
%files = dir(fullfile(folder, '*.wav'));
%X = 59;
%[audio, fs] = audioread(fullfile(folder, files(X).name));
%audio = audio(:,1);

%% הרצת אלגוריתם פרוני והצגת תוצאות
% 1. הגדרת פרמטרים
%win_len_sec = 0.030; % חלון של 30 מילישניות

% 2. קריאה לפונקציה שבנית
%[freq_map, amp_map, t_vec] = prony_tracker(audio, fs, win_len_sec);

% 3. תצוגה גרפית (Visualization)
%figure('Name', 'Prony Frequency Tracking', 'Color', 'w');

% נציג את התדרים כנקודות על הגרף.
% ציר ה-X הוא זמן, ציר ה-Y הוא תדר.
% הנקודות מייצגות את 8 התדרים שנמצאו בכל חלון זמן.
%plot(t_vec, freq_map, '.', 'MarkerSize', 8);

%title(['Prony Method: Top 8 Frequencies (Window: ' num2str(win_len_sec*1000) 'ms)']);
%%xlabel('Time (seconds)');
%ylabel('Frequency (Hz)');
%grid on;
%ylim([0 fs/2]); % הצגת תחום התדרים עד מחצית תדר הדגימה
%xlim([0 max(t_vec)]);

% אופציונלי: הוספת מקרא כדי להבין איזה צבע הוא איזה "דירוג" של תדר (הראשון הכי חזק וכו')
%legend('Freq 1 (Strongest)', 'Freq 2', 'Freq 3', 'Freq 4', ...
 %      'Freq 5', 'Freq 6', 'Freq 7', 'Freq 8', 'Location', 'bestoutside');