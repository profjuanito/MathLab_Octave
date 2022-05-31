function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

for ii=1:m
    z(ii,1) = theta'*X(ii,:)';
end
g = sigmoid(z);

temp0 = 0.0;
for iteration = 1:m
    %printf('iteration = %d  g(iteration) = %f    temp0 = %f \n', iter, g(iteration), temp0);
    temp0 = temp0 + ( -y(iteration)*log( g(iteration) ) - ( (1 - y(iteration) ) * log( 1 - g(iteration) ) ) );
end

temp1 = 0.0;
for iteration = 2:num_features
    temp1= temp1 + theta(iteration)^2;
end
J = (1/m)*temp0 + (lambda/(2*m))*temp1;



for itera=1:num_features
    temp(itera)=0.0;

    for it = 1:m
        %printf('it = %d \n',it);
        temp(itera) = temp(itera) + ( (g(it) - y(it))*X(it,itera) );
    end

end


theta(1) = ((1)/m)*temp(1);

for itera = 2:num_features
    %printf('itera = %d \n',itera);
    theta(itera) = (1/m)*temp(itera) + (lambda/m)* theta(itera) ;
end

grad=theta;



% =============================================================

end
