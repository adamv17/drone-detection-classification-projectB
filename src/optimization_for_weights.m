
        function [omega_star]=optimize_weights(data)
            % ==========================================
            % 1. LOAD DATA
            % ==========================================
            % 1. Get N (Number of samples/rows)
            N = size(data, 1);

            % 2. Get C (Number of classes)
            % Total columns minus 1 (because the first column is the real class)
            C = size(data, 2) - 1;
            
            % 3. Get D (Number of grades)
            % Look inside the first grade vector to see how long it is
            D = length(data{1, 2}); 
            
            % 4. Extract the real classes vector
            y_labels = cell2mat(data(:, 1));
            
            % 5. Extract and reshape the grades into the N x C x D matrix
            flat_grades = cell2mat(data(:, 2:end));
            g_data = reshape(flat_grades, N, C, D);
            
            % ==========================================
            % 2. INITIALIZE WEIGHTS VECTOR
            % ==========================================
            
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


