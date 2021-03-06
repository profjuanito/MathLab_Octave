function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(2, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    temp1 = 0;
    temp2 = 0;
    for itera=1:m
        %  itera
        temp1 = temp1 + ( ( theta(1)*X(itera,1) + theta(2)*X(itera,2) ) - y(itera) ) * X(itera,1);
        %printf("temp1 =");
        %  temp1
        temp2 = temp2 + ( ( theta(1)*X(itera,1) + theta(2)*X(itera,2) ) - y(itera) ) * X(itera,2);
        %printf("temp2 =");
        %  temp2
    end
    theta(1) = theta(1) - (alpha/m)*temp1;
    theta(2) = theta(2) - (alpha/m)*temp2;
    %printf("theta(1) =");
    %theta(1)
    %printf("theta(2) =");
    %theta(2)
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %printf("J_history (%d) = %f \n", iter, J_history(iter));
end

end
