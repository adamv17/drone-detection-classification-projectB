%% 1. Setup and Load Audio
clear; clc; close all;

dataPath = '../../datasets/Drone-detection-dataset-master/Data/Audio'; 
wavFiles = dir(fullfile(dataPath, '*.wav'));

% --- TEST: Choose a file ---
numFile = 59; 
filePath = fullfile(wavFiles(numFile).folder, wavFiles(numFile).name);
fprintf('Processing file: %s\n', wavFiles(numFile).name);

[x, fs] = audioread(filePath);
if size(x, 2) > 1, x = x(:,1); end 

% --- PARAMETERS ---
% 0.5s is a good balance for moving drones
duration = 0.5; 
numSamples = min(length(x), floor(duration * fs));
x = x(1:numSamples); 

%% 2. Compute Bicoherence
nfft = 512;       
window = hamming(nfft);
noverlap = floor(length(window) * 0.5); 

% A. Segment
X_segments = buffer(x, nfft, noverlap, 'nodelay');
X_segments = X_segments .* window; 
X_fft = fft(X_segments, nfft); 
X_fft = X_fft(1:nfft/2+1, :); 
[numFreqs, numSegs] = size(X_fft);

fprintf('Number of Segments (K): %d\n', numSegs);

% B. Vectorized Calculation
Bispectrum = zeros(numFreqs, numFreqs);
P_spec = zeros(numFreqs, 1);

for k = 1:numSegs
    Xk = X_fft(:, k); 
    XX_mat = Xk * Xk.'; 
    
    [I, J] = meshgrid(1:numFreqs, 1:numFreqs);
    sum_idx = I + J - 1; 
    valid_mask = sum_idx <= numFreqs;
    
    X_sum_conj = zeros(numFreqs, numFreqs);
    X_sum_conj(valid_mask) = conj(Xk(sum_idx(valid_mask)));
    
    Bispectrum = Bispectrum + (XX_mat .* X_sum_conj);
    P_spec = P_spec + abs(Xk).^2;
end

Bispectrum = Bispectrum / numSegs;
P_spec = P_spec / numSegs;

% C. Normalization
[P1, P2] = meshgrid(P_spec, P_spec);
P_sum = zeros(numFreqs, numFreqs);
P_sum(valid_mask) = P_spec(sum_idx(valid_mask));
denom = (P1 .* P2) .* P_sum;

% Calculate Squared Bicoherence
bic2 = (abs(Bispectrum).^2) ./ (denom + 1e-12);

%% 3. The RIGOROUS Statistical Mask
% We DO NOT mask by amplitude. We mask by Probability.

% Haubrich's Threshold for 95% Confidence
% "If the signal was pure noise, 95% of bicoherence values would be below this line"
significance_threshold = 3 / numSegs; 
fprintf('Statistical Significance Threshold (95%%): %.4f\n', significance_threshold);

freq_axis = linspace(0, fs/2, numFreqs);
[F1, F2] = meshgrid(freq_axis, freq_axis);

% Mask 1: Geometry (Golden Triangle)
geom_mask = (F2 <= F1) & ((F1 + F2) <= fs/2);

% Mask 2: Statistical Significance
% Only keep pixels that are statistically unlikely to be noise
stat_mask = bic2 > significance_threshold;

% Combined Mask
final_mask = geom_mask & stat_mask;

% Extract Valid Pixels
valid_pixels = bic2(final_mask);

if isempty(valid_pixels)
    valid_pixels = 0; % No significant coupling found
end

%% 4. Visualization
figure('Position', [100, 100, 1000, 400], 'Color', 'w');

subplot(1, 2, 1);
% Plot the raw bicoherence, but draw a contour at the significance level
plot_bic = bic2;
plot_bic(~geom_mask) = NaN; 
pcolor(freq_axis, freq_axis, plot_bic); shading flat;
colormap('jet'); colorbar; 
clim([0 0.5]); 
hold on;
% Draw the "Significance Line" - anything inside is valid
contour(freq_axis, freq_axis, bic2 > significance_threshold, 1, 'w', 'LineWidth', 1.5);
title(['Bicoherence (White Line = ' num2str(significance_threshold, '%.3f') ' threshold)']);
xlabel('Hz'); ylabel('Hz');

subplot(1, 2, 2);
% Compare the histogram against the threshold
histogram(valid_pixels, 40, 'BinLimits', [0, 1], 'Normalization', 'probability');
xline(significance_threshold, 'r--', 'LineWidth', 2, 'Label', 'Noise Floor');
title('Histogram of SIGNIFICANT Coupling');
xlabel('Coupling Strength');
grid on;

sgtitle(['Statistical Analysis (K=' num2str(numSegs) ')'], 'Interpreter', 'none');