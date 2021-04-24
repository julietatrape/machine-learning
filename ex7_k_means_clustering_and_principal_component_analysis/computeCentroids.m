function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

all_together=[X double(idx)];
sum_mu = zeros(K, n+1);
mean_mu = zeros(K, n+1);

for j=1:K,
   sum_mu(j,:)=sum(all_together((all_together(:,n+1)==j),:));
    if sum_mu(j,n+1)~=0,
    mean_mu(j,:)=sum_mu(j,:)./(sum_mu(j,:)(n+1)/j);
    %else
    %mean_mu(j,:)=zeros(n+1);
    end
   centroids=mean_mu(:,1:n);
end


% =============================================================


end

