clc 
clear
close all
%% pSet, qSet: 3xNPts

% pSet - ref da CAM - size(3,N)
% qSet - ref do KUKA - size(3,N)
[pSet, qSet] = getDados();  


% Transformation from Camera to KUKA
[R_CAM_KUKA,t_CAM_KUKA] = rigidBodyTransformationSVD(pSet, qSet)

pTest = pSet;

qTest = R_CAM_KUKA*pTest + t_CAM_KUKA;
q2Excel=qTest'
qVal = qSet;

erro = abs(qTest-qVal)
% X Y Z
erroSTATS = zeros(3,3);
erroSTATS(1,1) = min(erro(1,:)); erroSTATS(1,2) = max(erro(1,:));erroSTATS(1,3) = mean(erro(1,:));
erroSTATS(2,1) = min(erro(2,:)); erroSTATS(2,2) = max(erro(2,:));erroSTATS(2,3) = mean(erro(2,:));
erroSTATS(3,1) = min(erro(3,:)); erroSTATS(3,2) = max(erro(3,:));erroSTATS(3,3) = mean(erro(3,:));
erroSTATS = erroSTATS';


for i = 1:size(pSet,2)
    RMSE(i) = sqrt( erro(1,i)^2 + erro(2,i)^2 + erro(3,i)^2 );
end
Erro = RMSE
% figure
% bar(Erro)

[maxdiff,maxIndex] = max(RMSE);
maxdiff
maxIndex = maxIndex -1 

[mindiff,minIndex] = min(RMSE);
mindiff
minIndex = minIndex -1 

mean_error = mean(RMSE)

variance = var(RMSE)
%% Desired position

pDesired = [-37.08	107.91	551.04]';

qDesired = R_CAM_KUKA*pDesired + t_CAM_KUKA;
qDesired = qDesired'

