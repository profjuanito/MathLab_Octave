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
grad = theta;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%alpha = 0.0015; num_iters =500000; %-14.024578/0.117351/0.111440  %error tempo 7171.84   125.85 / 119.51    %algoritmo nuevo
%alpha = 0.0015; num_iters =1000000; %-17.660467/0.146316/0.140770  %error tempo 13527   125.46 / 120.70    %algoritmo nuevo

%alpha = 0.0015; num_iters =1000000; %-17.660467/0.146316/0.140770  %error tempo 8527.74   125.46 / 120.70    %algoritmo nuevo
%alpha = 0.0015; num_iters =300000;  %-11.479568 /0.097151/0.091021 %error tempo 2740.06   126.12 / 118.16    %algoritmo nuevo
%alpha = 0.0015; num_iters =100000;  %-6.876278 /0.061058/0.054455  %error tempo 857.569   126.28 / 112.62    %algoritmo nuevo
%alpha = 0.0015; num_iters =1000000; %-17.660467/0.146316/0.140770  %error tempo 5466.08   125.46 / 120.70    %algoritmo anterior
%alpha = 0.0015; num_iters =1000000; %-17.660467/0.146316/0.140770  %error tempo 8527.74   125.46 / 120.70    %algoritmo nuevo
%alpha = 0.0015; num_iters =300000;  %-11.479568 /0.097151/0.091021 %error tempo 2740.06   126.12 / 118.16    %algoritmo nuevo
%alpha = 0.0015; num_iters =100000;  %-6.876278 /0.061058/0.054455  %error tempo 857.569   126.28 / 112.62    %algoritmo nuevo
%alpha = 0.0015; num_iters =1000000; %-17.660467/0.146316/0.140770  %error tempo 5466.08   125.46 / 120.70    %algoritmo anterior

%                                 teste1         teste2
alpha = 0.0015;     % 0.0015;        %1           %0.1
num_iters = 1000000;    % 300000;    %1           %2

for ii=1:m
    z(ii,1) = grad'*X(ii,:)';
end
g = sigmoid(z);

temp0 = 0.0;
for iteration = 1:m
    %printf('iteration = %d  g(iteration) = %f    temp0 = %f \n', iter, g(iteration), temp0);
    temp0 = temp0 + ( -y(iteration)*log( g(iteration) ) - ( (1 - y(iteration) ) * log( 1 - g(iteration) ) ) );
end
J = (1/m)*temp0;


for iter = 1:num_iters
    %printf('num_iters = %d \n',iter);
    temp = zeros(size(theta));

    for it = 1:num_features
        temp(it) = sum( (sigmoid(grad'*X') - y').*X(:,it)');
    end
                                  
    for it = 1:num_features
        grad(it) = grad(it) - (alpha/m)*temp(it);
    end

end
                       
grad

% =============================================================

end
                       
                       
                       
                       
                       for ii=1:m
                       z(ii,1) = theta'*X(ii,:)';
                       end
                       g = sigmoid(z);
                       
                       
                       for itera=1:num_features
                       temp(itera)=0.0;
                       
                       for it = 1:m
                       %printf('it = %d \n',it);
                       temp(itera) = temp(itera) + ( (g(it) - y(it))*X(it,itera) );
                       end
                       %temp(itera)
                       
                       end
                       
                       for itera = 1:num_features
                       %printf('itera = %d \n',itera);
                       theta(itera) = theta(itera) - (alpha/m)*temp(itera);
                       end                       
                       
                       
                       
                       
                       
                       
                       
