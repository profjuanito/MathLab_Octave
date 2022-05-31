function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

for ii=1:m
    z(ii,1) = theta'*X(ii,:)';  % 1,3  3,100
end
g = sigmoid(z);

temp0 = 0.0;
for iteration = 1:m
    %printf('iteration = %d  g(iteration) = %f    temp0 = %f \n', iter, g(iteration), temp0);
    temp0 = temp0 + ( -y(iteration)*log( g(iteration) ) - ( (1 - y(iteration) ) * log( 1 - g(iteration) ) ) );
end
J = (1/m)*temp0;


for itera=1:num_features
    temp(itera)=0.0;

    for it = 1:m
        %printf('it = %d \n',it);
            temp(itera) = temp(itera) + ( (g(it) - y(it))*X(it,itera) );
    end

end
                                  
for itera = 1:num_features
    %printf('itera = %d \n',itera);
    theta(itera) = (1/m)*temp(itera);
end

grad=theta;


% =============================================================

end
