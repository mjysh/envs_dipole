close all;
clear;
%% NN setup
Nobs = 5;
Naction = 1;
N1 = 128;
N2 = 128;
NNsize = [N1, N2];
policy_path = 'bestpolicy_lab';

% ID = fopen('bestpolicy_egograd/agent_00_net_weights.raw');
% Params = fread(ID,'float');
% ID = fopen('bestpolicy_egograd/agent_00_scaling.raw');
% SCALE = fread(ID,'double');
% obs_mean = SCALE(1:5);
% obs_scale = SCALE(6:10);

%% load policy
vracerNN = loadvracerNN(policy_path,Nobs,N1,N2,Naction);

policy = @(o) obs_to_act(o, vracerNN);

%% load CFD Data
CFD = load('CFDData.mat');
%% simulation initial conditions
pos0 = [-12, -2.15, pi];
target = [-12, 2.15];


%%
while ~done
    obs = 
end
[flowU, flowV, ~]= adapt_time_interp(CFD,time+t,z(1),z(2));
activation = @tanh;

% load('/home/yusheng/navigation_envs/dipole_new/plotTools/bestpolicy_lab/trajectory12-12.mat','observations','actions','states','reward','target')
% load('/home/yusheng/navigation_envs/dipole_new/plotTools/lab/trajectory.mat')
% load('/home/yusheng/navigation_envs/dipole_new/plotTools/bestpolicy_egograd/trajectory12-12.mat')

action_from_policy = zeros(size(actions),'like',actions);
history = zeros(128,length(actions));
for k = 1:length(actions)
obs = observations(k,:)';
% obs = [0
% 4.3
% -3.14
% -0.932246
% -0.0434474];
% actionsAndValue = end_to_end(obs);
% 
% actionsAndValue = obs_to_act(obs, vracerNN);
obs = (obs-vracerNN.obs_mean).*vracerNN.obs_scale;
s1 = vracerNN.W1*obs + vracerNN.B1;
i1 = activation(s1);

s2 = vracerNN.W2*i1 + vracerNN.B2;
i2 = activation(s2);

s_res = vracerNN.W_res.*i1 + vracerNN.B_res;
i_res = s_res + i2;

s3 = vracerNN.W3*i_res + vracerNN.B3;
actionsAndValue = tanh(s3);
action_from_policy(k) = actionsAndValue(2);
history(:,k) = i_res;
end

% figure,plot(action_from_policy);
% hold on;
% plot(actions)
% function actionsAndValue = obs_to_act(obs, obs_mean, obs_scale, W1, B1, activation, W2, B2, W_res, B_res, W3, B3)
% 
% obs = (obs-obs_mean).*obs_scale;
% s1 = W1*obs + B1;
% i1 = activation(s1);
% 
% s2 = W2*i1 + B2;
% i2 = activation(s2);
% 
% 
% s_res = W_res.*i1 + B_res;
% i_res = s_res + i2;
% 
% s3 = W3*i_res + B3;
% actionsAndValue = tanh(s3);
% end