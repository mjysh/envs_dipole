close all;
clear;
%%
figureDefaultSettings;
%%
traj_path = '/home/yusheng/navigation_envs/dipole_new/plotTools/bestpolicy_lab/trajectory12-12.mat';
policy_path = './bestpolicy_lab';
load(traj_path);

Nobs = 5;
Naction = 1;
N1 = 128;
N2 = 128;
NNsize = [N1, N2];
vracerNN = loadvracerNN(policy_path,Nobs,N1,N2,Naction);

fun = @(o) obs_to_act(o, vracerNN);
% A = rand(2,Nobs);
% fun = @(x) A*x;
for N = 1:length(observations)-1
    % N = 361
    obs1 = observations(N,:)';
    obs2 = observations(N+1,:)';
    % obs1 = [1;1;1;1;1];
    % obs2 = [5;2;2;5;2];

    out1 = fun(obs1);
    out2 = fun(obs2);
    realChange = out2 - out1;
    dado = backward_matrix(obs1,vracerNN);
    obs_diff = obs2-obs1;
    linearChange = dado*obs_diff;
    linearErr = linearChange - (out2 - out1);
    % partial = zeros(2,Nobs);
    % for i = 1:Nobs
    %     temp = obs1;
    %     temp(i) = obs2(i);
    %     partial(:,i) = fun(temp) - out1;
    % end
    partial = zeros(2,2);
    temp = obs1;
    temp(1:2) = obs2(1:2);
    partial(:,1) = fun(temp) - out1;
%     temp = obs1;
%     temp(3) = obs2(3);
%     partial(:,2) = fun(temp) - out1;
    temp = obs1;
    temp(3:5) = obs2(3:5);
    partial(:,2) = fun(temp) - out1;

    uncoupleErr = sum(partial,2) - realChange;
    relLinearErr(N) = linearErr(2)/realChange(2);
    relUncoupleErr(N) = uncoupleErr(2)/realChange(2);
end