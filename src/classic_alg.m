close all

%% FFT
function fft_filtered_transform(audio, fs ,K)
    N = length(audio);
    Y = fft(audio);
    f = (0:N-1) * (fs/N);
    
    baseline = medfilt1(abs(Y), 101);
    mean_fft_magnitude=mean(abs(Y));
    threshold = baseline * K;
    
    M = Y;
    M(abs(M) < threshold) = 0;
    % הצגת ספקטרום
    
    figure;
    plot(f, abs(M));
    title('FFT Magnitude');
    xlabel('Frequency (Hz)');
    ylabel('|Y(f)|');
    xlim([0 fs/2]);      % תדרים חיוביים
    grid on;
end
%% --- STFT ---
function stft_filtered_transform(audio, fs, K)
    window = 4096;        
    overlap = window / 2;
    nfft = 4096;
    
    % מחשבים את ה‑STFT
    [S,F,T] = spectrogram(audio, window, overlap, nfft, fs);
    
    % המרה למגניטודה
    S_mag = abs(S);
    
    % --- סינון Adaptive Median Threshold ---
    baseline = medfilt1(S_mag, 5, [], 1);  % חצי-ספקטרום לפי תדר (axis=1)
    threshold = baseline * K;
    
    % יוצרים מסכה
    mask = S_mag >= threshold;
    S_filtered = S_mag .* mask;
    
    % --- הצגת ספקטרוגרם מסונן ---
    figure;
    imagesc(T, F, 20*log10(S_filtered)); % dB
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('STFT (Filtered Spectrogram)');
    colorbar;
end

%%PSD
function [PSD,F]=PSD_transform(audio, fs, K)
    NFFT = 4096; 
    window = hamming(NFFT); 
    noverlap = NFFT / 2;
    
    
    [pxx, f] = pwelch(audio, window, noverlap, NFFT, fs);
    
    
    figure;
    plot(f, 10*log10(pxx)); % המרה לדציבלים
    grid on;
    title(['Power Spectral Density '], 'Interpreter', 'none');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0 20000]); 
    PSD=10*log10(pxx);
    F=f;

end

%%alg

folder = "C:\Users\yahal\OneDrive\מסמכים\GitHub\drone-detection-classification-projectB\datasets\Drone-detection-dataset-master\Data\Audio";   % כאן שים את הנתיב לתיקייה
files = dir(fullfile(folder, '*.wav'));

X = 59;
[audio, fs] = audioread(fullfile(folder, files(X).name));
audio = audio(:,1);
sound(audio);
fft_filtered_transform(audio,fs,1.7);
stft_filtered_transform(audio,fs,1.7);
[psd,f]=PSD_transform(audio,fs);


[pks, locs] = findpeaks(psd, f, 'MinPeakDistance', 200);

figure;
plot(f, psd); hold on;
plot(locs, pks, 'ro', 'MarkerSize', 8);
xlabel('Frequency (Hz)');
ylabel('PSD (dB/Hz)');
title('PSD Peaks');
grid on;