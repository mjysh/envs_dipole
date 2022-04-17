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
policyType = 'egoGradLR';
% dir = './lab/';
dir = ['./' policyType '/'];
% [target, init, X, Y, Theta, time, reward] = getTrajectory(dir);


%%

% policyType = 'lab';
load([dir 'trajectory.mat']);
time = time(1:end-1)';
dx = observations(1:end-1,1);
dy = observations(1:end-1,2);
u = observations(1:end-1,3);
v = observations(1:end-1,4);
dudy = observations(1:end-1,5);
dvdy = observations(1:end-1,6);
uratio = v./u;
xratio = dy./dx;
xangle = atan2(dy,dx);
uangle = atan2(v,u);
gradratio = dvdy./dudy;
action_traj = actions;
fig1 = figure('Position',[105 414 2420 465]);
left_color = [0 0 0];
right_color = [.5 .0 0];
set(fig1,'defaultAxesColorOrder',[left_color; right_color]);
yyaxis left
plot(time,action_traj,'k'); hold on;
yyaxis right
colororder('default')
plot(time,dx,'--', time,dy,'--', time,u,'--', time,v,'--');
% plot(time,dx,'--', time,dy,'--', time,u,'--', time,v,'--', time,dudy,'--', time,dvdy,'--');
legend({'action ($\dot\theta$)','$\Delta x_1$','$\Delta x_2$','$u$','$v$'},Interpreter="latex")
% legend({'action ($\dot\theta$)','$\Delta x_1$','$\Delta x_2$','$u$','$v$','$\partial_2 u$','$\partial_2 v$'},Interpreter="latex")
%%
v = VideoWriter([policyType '_trajWithPolicy.avi']);
open(v);
figure('Position',[960 848 921 850]); 
imname = ['/home/yusheng/CFDadapt/Movie/movie',num2str(round(mod(time(1),4.5)*20),"%04d")];
[bg,map] = imread(imname,"png");
% flow field
subplot(2,1,1); hold on;
im = image([-24,8],[-8,8],ind2rgb(flipud(bg),map));
axis equal;

X = states(:,1);
Y = states(:,2);
Theta = states(:,3);
plot(X(1),Y(1),'ko');
plot(target(1),target(2),'kp');
xlim([-23.5,0]);
ylim([-6,6]);
% policy
subplot(2,1,2); hold on;
load([policyType '1CFD_policy-9/policyTest' num2str(round(mod(time(1)*20,90))/20,'%.2f') '.mat'],'actions');
colormap parula
theta_index = (wrapToPi(Theta(1))+pi)/(pi/18) + 1;
index_down = floor(theta_index);
index_up = index_down + 1;
w_up = index_up - theta_index;
w_down = theta_index - index_down;
actionAtAngle = w_down*reshape(actions(index_down:36:end),46,[]) + w_up*reshape(actions(mod(index_up-1,36)+1:36:end),46,[]);


impolicy = imagesc([-23,-1],[-5,5.25],actionAtAngle,'AlphaData',0.8,[-1,1]);
plot(X(1),Y(1),'ko');
plot(target(1),target(2),'kp');
axis equal;
title([num2str(Theta(1)), num2str(theta_index)])
% colorbar("Ticks",[],'Location','westoutside')
cb = colorbar('Location','westoutside');
cb.Label.Interpreter = 'latex';
cb.Label.String = '$\dot\theta$';
xlim([-23.5,0]);
ylim([-6,6]);
frame = getframe(gcf);
writeVideo(v,frame);

for i = 1:length(time)
    %     pause(0.4)
    subplot(2,1,1);
    imname = ['/home/yusheng/CFDadapt/Movie/movie',num2str(round(mod(time(i),4.5)*20),"%04d")];
    [bg,map] = imread(imname,"png");
    im.CData = ind2rgb(flipud(bg),map);
    plot(X(i),Y(i),'k.');
    subplot(2,1,2);
    plot(X(i),Y(i),'k.');
    load([policyType '1CFD_policy-9/policyTest' num2str(mod(round(time(i)*20), 90)/20,'%.2f') '.mat'],'actions');
    theta_index = (wrapToPi(Theta(i))+pi)/(pi/18) + 1;
    index_down = floor(theta_index);
    index_up = index_down + 1;
    w_up = index_up - theta_index;
    w_down = theta_index - index_down;
    actionAtAngle = w_down*reshape(actions(index_down:36:end),46,[]) + w_up*reshape(actions(mod(index_up-1,36)+1:36:end),46,[]);
    impolicy.CData = actionAtAngle;
    title([num2str(Theta(i)),'to', num2str(theta_index)])
    drawnow;
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v);
video_conversion([filename '_trajWithPolicy.avi']);
%% convert to mp4 and delete the original file
function video_conversion(filename)
system(['ffmpeg -i ' filename '_trajWithPolicy.avi' ...
    ' -c:v libx264 -profile:v high -crf 20 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" ' ...
    '-pix_fmt yuv420p ' filename 'output.mp4 -y']);
system(['[ -f' filename 'output.mp4 ] && rm ' filename '_trajWithPolicy.avi']);
end