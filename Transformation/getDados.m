function [pSet,qSet] = getDados()
%% CLOSE SETUP

% % pSet - ref da CAM
% pSet = readmatrix('CLOSE_CAM_ref.csv');
% % qSet - ref do KUKA
% qSet = readmatrix('CLOSE_KUKA_ref.csv');

%% MID SETUP

% % pSet - ref da CAM
% pSet = readmatrix('MID_CAM_ref.csv');
% % qSet - ref do KUKA
% qSet = readmatrix('MID_KUKA_ref.csv');

%% FAR SETUP

% pSet - ref da CAM
pSet = readmatrix('FAR_CAM_ref.csv');
% qSet - ref do KUKA
qSet = readmatrix('FAR_KUKA_ref.csv');

%%
pSet = pSet(:,2:4)';
qSet = qSet(:,2:4)';
end
