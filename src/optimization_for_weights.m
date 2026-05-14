
   
classdef optimization_for_weights
    methods (Static)
        function [omega_star]=optimize_weights()
            % ==========================================
            % 1. LOAD YOUR REAL DATA
            % ==========================================
            % Replace this block with your actual data loading mechanism.
            % Example: load('my_dataset.mat', 'g_data', 'y_labels');
            
            % For this example, let's assume you have loaded:
            % g_data: A 3D array of size [N, 5, 4] containing your features
            % y_labels: A column vector of size [N, 1] containing correct classes (1-5)
            
            % (Simulating the loaded data here so the script runs)
            N_loaded = 850; % Imagine your trainset has 850 points
            g_data = randn(N_loaded, 5, 4); 
            y_labels = randi([1, 5], N_loaded, 1);
            
            % ==========================================
            % 2. DERIVE DIMENSIONS DYNAMICALLY
            % ==========================================
            % Extract N, C, and D directly from your loaded data
            [N, C, D] = size(g_data);
            
            % Verify dimensions just to be safe
            if C ~= 5
                error('Expected 5 classes, but found %d', C);
            end
            if D ~= 4
                error('Expected feature dimension D to be 4, but found %d', D);
            end
            
            disp(['Training on N = ', num2str(N), ' samples with D = ', num2str(D)]);
            
            % Initialize the 4x1 weight vector
            omega_init = randn(D, 1) * 0.01;
            
            % ==========================================
            % 3. CONFIGURE AND RUN OPTIMIZATION
            % ==========================================
            options = optimoptions('fminunc', ...
                'SpecifyObjectiveGradient', true, ...
                'Display', 'iter', ...
                'Algorithm', 'trust-region');
            
            % Define the objective, passing in your real data
            objective_func = @(w) custom_cross_entropy(w, g_data, y_labels);
            
            % Solve for the optimal 4x1 weight vector
            [omega_star, final_loss] = fminunc(objective_func, omega_init, options);
            
            disp('Optimization Complete!');
            disp('Optimal Weights (omega_star):');
            disp(omega_star);
            
            
            % =============================
            % 4. THE OBJECTIVE FUNCTION 
            % =============================
            function [loss, grad] = custom_cross_entropy(omega, g, y)
                [N, C, D] = size(g);
                
                loss = 0;
                grad = zeros(D, 1);
                lambda = 1e-4; % L2 Regularization
                
                for i = 1:N
                    % Get the 5x4 feature matrix for this specific data point
                    g_i = squeeze(g(i, :, :)); 
                    
                    % Calculate raw scores for all 5 classes (5x1 vector)
                    scores = g_i * omega; 
                    
                    % Softmax with numerical stability
                    max_score = max(scores);
                    exp_scores = exp(scores - max_score);
                    sum_exp = sum(exp_scores);
                    probs = exp_scores / sum_exp; 
                    
                    % 1. Accumulate Loss
                    correct_class = y(i);
                    loss = loss - log(probs(correct_class));
                    
                    % 2. Accumulate Gradient
                    expected_g = g_i' * probs; 
                    actual_g = g_i(correct_class, :)'; 
                    
                    grad = grad + (expected_g - actual_g);
                end
                
                % Add regularization
                loss = loss + (lambda / 2) * sum(omega.^2);
                grad = grad + lambda * omega;
            end
        end
    end

end
