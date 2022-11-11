close all;
clear;
%%
figureDefaultSettings;
%% NN setup
Nobs_geo = 5;
Nobs_ego = 6;
Naction = 1;
N1 = 128;
N2 = 128;
NNsize = [N1, N2];
policy_path_ego = '/home/yusheng/smarties/apps/dipole_adapt/paper/egoLRGrad8';
% policy_path_geo = '/home/yusheng/smarties/apps/dipole_adapt/paper/lab1';
policy_path_geo = '/home/yusheng/navigation_envs/dipole_new/tools/bestpolicy_geo';
%% load policy
vracerNN_ego = loadvracerNN(policy_path_ego,Nobs_ego,N1,N2,Naction);
policy_ego = @(o) obs_to_act(o, vracerNN_ego);
vracerNN_geo = loadvracerNN(policy_path_geo,Nobs_geo,N1,N2,Naction);
policy_geo = @(o) obs_to_act(o, vracerNN_geo);
%%
% load('/home/yusheng/navigation_envs/dipole_new/tools/egoGradLR/trajectory2.mat', 'states')
load('/home/yusheng/navigation_envs/dipole_new/tools/bestpolicy_geo/trajectory12-12.mat','states','observations','reward','target','time')
obs_geo = zeros(length(states),5);
obs_ego = zeros(length(states),6);
action_ego = zeros(length(time),1);
action_geo = zeros(length(time),1);
CFD = load('CFDData.mat');
%%
for i = 1:length(states)
    obs_geo(i,:) = observation_geo(states(i,:),time(i),CFD,target);
    obs_ego(i,:) = observation_ego(states(i,:),time(i),CFD,target);
    %     [action_ego(i),~] = policy_ego(obs_ego(i,:));
    [action_geo(i),~] = policy_geo(obs_geo(i,:));
end
%%
perAngleTest(policy_geo,@observation_geo,CFD,-11.97,0.50,1,target)
perAngleTest(policy_ego,@observation_ego,CFD,-11.97,0.50,1,target)
%%
figure, plot(action_ego);
hold on;
plot(action_geo)

%%
c = colororder;
for N = 1:10:450
    test = states(N,:);
    orts = -pi:pi/360:pi;
    action_geo_angle = zeros(length(orts),1);
    action_ego_angle = zeros(length(orts),1);
    for i = 1:length(orts)
        test(3) = orts(i);
        obs_geo = observation_geo(test,time(N),CFD,target);
        obs_ego = observation_ego(test,time(N),CFD,target);
        [action_geo_angle(i),~] = policy_geo(obs_geo);
        [action_ego_angle(i),~] = policy_ego(obs_ego);
    end
    figure;
    plot(orts, action_geo_angle,'Color',c(1,:)); hold on
    plot(orts, action_ego_angle,'Color',c(2,:));
    plot(orts,zeros(length(orts),1),'k--')
%     plot(states(N,3),actions(N),'k*')
    zeros_geo = find(action_geo_angle(1:end-1).*action_geo_angle(2:end)<0 & action_geo_angle(1:end-1) > 0);
    zeros_ego = find(action_ego_angle(1:end-1).*action_ego_angle(2:end)<0 & action_ego_angle(1:end-1) > 0);
    for k = 1:length(zeros_ego)
        plot(ones(10,1)*mean(orts(zeros_ego(k):zeros_ego(k)+1)),linspace(-1,1,10),'--','Color',c(2,:));
    end
    for k = 1:length(zeros_geo)
        plot(ones(10,1)*mean(orts(zeros_geo(k):zeros_geo(k)+1)),linspace(-1,1,10),'--','Color',c(1,:));
    end
end
%% target direction in geocentric policy
t_cfd = 0;
[X, Y, U, V, U_tot, V_tot,X_multi, Y_multi, U_multi,V_multi] = getTargetDirection(CFD, target, policy_geo,@observation_geo, t_cfd);
% [X, Y, U, V, U_tot, V_tot,X_multi, Y_multi, U_multi,V_multi] = getTargetDirection(CFD, target, policy_ego,@observation_ego, t_cfd);
% [X, Y, U, V] = getTargetDirection(CFD, target, policy_geo,@observation_geo, t_cfd);
%%
figure('Position',[965 687 1061 635]);
[bg,map] = imread(['/home/yusheng/CFDadapt/Movie_visit/movie' num2str(t_cfd*20,'%04.f') '.png'],"png");hold on;
image([-24,8],[8,-8],ind2rgb(bg,map));
quiver(X,Y,U,V,'AutoScale','off') % single stable fixed points
axis equal;
hold on
% quiver(X,Y,U_tot,V_tot,'AutoScale','off');
quiver(X_multi,Y_multi,U_multi,V_multi,'AutoScale','off','Color','b'); % multiple stable fixed points
scatter(X,Y,'.','MarkerEdgeColor',[0.5,0.5,0.5]);
% colorbar('Location','westoutside')
plot(target(1),target(2),'p',Color=[50/255,100/255,50/255]);
xlim([-23.5,0]);
ylim([-6,6]);

%%
function perAngleTest(policy,state_to_obs,CFD,x,y,t,target)
c = colororder;
test = [x,y,0];
orts = -pi:pi/360:pi;
action_angle = zeros(length(orts),1);
for i = 1:length(orts)
    test(3) = orts(i);
    obs = state_to_obs(test,t,CFD,target);
    [action_angle(i),~] = policy(obs);
end
plot(orts, action_angle,'Color',c(1,:)); hold on
plot(orts,zeros(length(orts),1),'k--')
roots = find(action_angle(1:end-1).*action_angle(2:end)<0 & action_angle(1:end-1) > 0);
for k = 1:length(roots)
    plot(ones(10,1)*mean(orts(roots(k):roots(k)+1)),linspace(-1,1,10),'--','Color',c(2,:));
end

end

function [X, Y, U, V, U_tot, V_tot,X_multi, Y_multi, U_multi,V_multi] = getTargetDirection(CFD, target, policy,state_to_obs, time)
bound_left = -23.5;
bound_right = -0.5;
bound_up = 5.5;
bound_down = -5.5;
[X,Y] = meshgrid(linspace(bound_left,bound_right,31), linspace(bound_down,bound_up,16));
U = zeros(size(X))*nan;
V = zeros(size(X))*nan;
U_tot = zeros(size(X))*nan;
V_tot = zeros(size(X))*nan;
U_multi = zeros(1000,1);
V_multi = zeros(1000,1);
X_multi = zeros(1000,1);
Y_multi = zeros(1000,1);
n_multi = zeros(1000,1);
for m = 1:size(X,1)
    for n = 1:size(X,2)
        x = X(m,n);
        y = Y(m,n);
        test = [x,y,0];
        orts = -pi:pi/360:pi;
        %         action_geo_angle = zeros(length(orts),1);
        action_angle = zeros(length(orts),1);
        for i = 1:length(orts)
            test(3) = orts(i);
            %             obs_geo = observation_geo(test,0,CFD,target);
            obs = state_to_obs(test,time,CFD,target);
            %             [action_geo_angle(i),~] = policy_geo(obs_geo);
            [action_angle(i),~] = policy(obs);
        end
        %         zeros_geo = find(action_geo_angle(1:end-1).*action_geo_angle(2:end) < 0 & action_geo_angle(1:end-1) > 0);
        roots = find(action_angle(1:end-1).*action_angle(2:end) < 0 & action_angle(1:end-1) > 0);
        %         if (isempty(zeros_geo))
        %             fprintf('not found at %4.2f, %4.2f\n',x,y)
        %         elseif (length(zeros_geo) == 1)
        %             ang = mean(orts(zeros_geo:zeros_geo+1));
        %             U(m,n) = cos(ang);
        %             V(m,n) = sin(ang);
        %         else
        %             fprintf('multiple values (%4.2f) at %4.2f, %4.2f\n',length(zeros_geo), x,y)
        %         end
        if (isempty(roots))
            fprintf('not found at %4.2f, %4.2f\n',x,y)
        elseif (length(roots) == 1)
            ang = mean(orts(roots:roots+1));
            U(m,n) = cos(ang)*0.8;
            V(m,n) = sin(ang)*0.8;
            [flowU, flowV, ~]= adapt_time_interp(CFD,time,x,y);
            U_tot(m,n) = U(m,n) + flowU;
            V_tot(m,n) = V(m,n) + flowV;
        else
            fprintf('multiple values (%4.2f) at %4.2f, %4.2f\n',length(roots), x,y)
            for k = 1:length(roots)
                n_multi = n_multi + 1;
                X_multi(n_multi) = x;
                Y_multi(n_multi) = y;
                ang = mean(orts(roots(k):roots(k)+1));
                U_multi(n_multi) = cos(ang)*0.8;
                V_multi(n_multi) = sin(ang)*0.8;
            end
        end
    end
end
X_multi = X_multi(1:n_multi);
Y_multi = Y_multi(1:n_multi);
U_multi = U_multi(1:n_multi);
V_multi = V_multi(1:n_multi);
end