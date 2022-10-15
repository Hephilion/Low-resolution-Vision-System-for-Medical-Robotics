function [R,t]=rigidBodyTransformationSVD(pSet, qSet)
% pSet, qSet: 3xNPts
% q = R*p+t
NPts = size(pSet,2);
% Centroids (weighted)
w = ones(1,NPts);
pSetAvg = mean(pSet,2);
qSetAvg = mean(qSet,2);

% Centered vectors
pCentered = pSet-repmat(pSetAvg,1,NPts);
qCentered = qSet-repmat(qSetAvg,1,NPts);

% Covariance matrix, S
W = diag(w);
S = pCentered*W*qCentered';

[U,~,V] = svd(S);
R = V*diag([ones(1,size(pSet,1)-1) det(V)*det(U)])*U';
t = qSetAvg-R*pSetAvg;
end