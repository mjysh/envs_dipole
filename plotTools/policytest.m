%%
close all;
clear;
%%
set(groot,'defaultLineLineWidth',1.5);
set(groot,'defaultFigureColor','w');
set(groot,'defaultTextFontsize',12);
set(groot,'defaultAxesFontsize',12);
set(groot,'defaultPolarAxesFontsize',12);
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultPolarAxesTickLabelInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultAxesLineWidth',1);
%%
load('/home/yusheng/navigation_envs/dipole_new/plotTools/egoGradLR/trajectory12-12.mat')
%% NN setup
Nobs = 6;
Naction = 1;
N1 = 128;
N2 = 128;
NNsize = [N1, N2];
policy_path = 'bestpolicy_egograd';
%% load policy
vracerNN = loadvracerNN(policy_path,Nobs,N1,N2,Naction);

policy = @(o) obs_to_act(o, vracerNN);
%%


figure("Position", [960 1061 363 252]);
% plot(targetX,targetY,'b*');
p = plot(obs,action,'.','MarkerSize',5);hold on
% axis equal;
% xlim([-24,0]);
% ylim([-10,210]);
ylabel('action');
xlabel('observation');