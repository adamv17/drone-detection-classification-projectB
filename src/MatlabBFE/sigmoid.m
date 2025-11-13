function y = sigmoid(x)
    % Standard logistic sigmoid function
    y = 1 ./ (1 + exp(-x));
end