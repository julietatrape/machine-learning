function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
steps=[C/100 sigma/10 C/10 sigma C sigma*10 C*10 sigma*100];
error=zeros(length(steps),length(steps));

for i=steps,
  C=i;
  for j=steps,
    sigma=j;
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    error(ind2sub(size(steps),find(steps==i)),ind2sub(size(steps),find(steps==j)))=mean(double(predictions ~= yval));
   end
end

minimo=min(min(error));
[k l] = ind2sub(size(error), find(error==minimo));
C=steps(k);
sigma=steps(l);


% =========================================================================

end
