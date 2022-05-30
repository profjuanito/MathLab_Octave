function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

num_features = size(X,2);
%theta


    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

for iter = 1:num_iters

    temp = zeros(1,num_features);

    for number_col = 1:num_features
        temp(number_col) = sum((X*theta - y).*X(:,number_col));
    end

    for number_col = 1:num_features
        theta(number_col) = theta(number_col) - (alpha/m)*temp(number_col);
    end

    %printf("theta(%d) =  %f   theta(%d) =  %f   theta(%d) =  %f  \n",number_col-2,theta(number_col-2),number_col-1,theta(number_col-1),number_col,theta(number_col));



    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);
    %printf("J_history (%d) = %f \n", iter, J_history(iter));
end

    % ============================================================

end

