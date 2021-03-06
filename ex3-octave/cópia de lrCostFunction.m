function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

num_features = rows(theta);
g = zeros(num_features,1);
z = zeros(m,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

for ii=1:m
    z(ii,1) = theta'*X(ii,:)';  %
end
g = sigmoid(z);

temp0 = 0.0;
for iteration = 1:m
    %printf('iteration = %d  g(iteration) = %f    temp0 = %f \n', iteration, g(iteration), temp0);
    temp0 = temp0 + ( -y(iteration)*log( g(iteration) ) - ( (1 - y(iteration) ) * log( 1 - g(iteration) ) ) );
end

temp1 = 0.0;
for iteration = 2:num_features
    temp1= temp1 + theta(iteration)^2;
end
J = (1/m)*temp0 + (lambda/(2*m))*temp1;


temp = zeros(size(theta));
temp = X'*(sigmoid(theta'*X') - y')';


grad(1) = ((1)/m)*temp(1);
for itera = 2:num_features
    %printf('itera = %d \n',itera);
    grad(itera) = (1/m)*temp(itera) + (lambda/m)* theta(itera) ;
end





% =============================================================

grad = grad(:);

end
