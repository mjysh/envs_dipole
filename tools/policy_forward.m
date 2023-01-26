close all;
clear;
%%
Nobs = 6;
Naction = 1;
N1 = 128;
N2 = 128;
NNsize = [N1, N2];
% ID = fopen('bestpolicy_egograd/agent_00_net_weights.raw');
% Params = fread(ID,'float');
% ID = fopen('bestpolicy_egograd/agent_00_scaling.raw');
% SCALE = fread(ID,'double');
% obs_mean = SCALE(1:5);
% obs_scale = SCALE(6:10);
policy_path = '/home/yusheng/smarties/apps/dipole_adapt/paper/egoLRGrad1';
vracerNN = loadvracerNN(policy_path,Nobs,N1,N2,Naction);
%%
vracerNN = loadvracerNN('bestpolicy_geo',Nobs,N1,N2,Naction);
% vracerNN = loadvracerNN('bestpolicy_egograd',Nobs,N1,N2,Naction);
% vracerNN = loadvracerNN('egoGradLR',Nobs,N1,N2,Naction);
%%
end_to_end = @(o) obs_to_act(o, vracerNN);

load('/home/yusheng/navigation_envs/dipole_new/plotTools/bestpolicy_geo/trajectory12-12.mat','observations','actions','states','reward','target')
% load('/home/yusheng/navigation_envs/dipole_new/plotTools/bestpolicy_egograd/trajectory12-12.mat')
% load('/home/yusheng/navigation_envs/dipole_new/plotTools/egoGradLR/trajectory12-12.mat')
figure,plot(states(:,1),states(:,2),'k');
ax1 = gca;hold on
[a,loc]=findpeaks(states(:,2));
loc = 395:397;
from = min(loc);
to = max(loc);
loc(end+1) = length(observations);
obs_min = min(observations(from:to,:));
obs_max = max(observations(from:to,:));
% obs_min = min(observations);
% obs_max = max(observations);
figure;
cm = hsv(length(loc));
for k = 1:length(loc)-1
testN = loc(k);
plot(ax1,states(testN,1),states(testN,2),'d','Color',cm(k,:))
obs_base = observations(testN,:);
for i = 1:Nobs
subplot(2,3,i)
baseoutput = end_to_end(observations(testN,:)');
plot(observations(testN,i),baseoutput(2),'d','Color',cm(k,:));
hold on;
obs_test = repmat(observations(testN,:)',[1,50]);
obs_test(i,:) = linspace(obs_min(i),obs_max(i),50);
testoutput = end_to_end(obs_test);
plot(obs_test(i,:),testoutput(2,:),'Color',cm(k,:));
end
end
figure,plot(actions(from:loc(end-1)));
%%

% softsign = @(x) x./(1+abs(x));
activation = @tanh;

end_to_end = @(o) obs_to_act(o, vracerNN);
% load('/home/yusheng/navigation_envs/dipole_new/plotTools/lab/trajectory.mat')
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