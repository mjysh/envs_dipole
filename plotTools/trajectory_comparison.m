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
% dir = './lab/';
dir = './egogradLR/';
[target, init, X, Y, Theta, time, reward] = getTrajectory(dir);


%%
policyType = 'egoGradLR';
% policyType = 'lab';
v = VideoWriter([policyType '_trajWithPolicy.avi']);
open(v);
figure('Position',[960 848 921 850]); 
imname = ['/home/yusheng/CFDadapt/Movie/movie',num2str(round(mod(time(1),4.5)*20),"%04d")];
[bg,map] = imread(imname,"png");
% flow field
subplot(2,1,1); hold on;
im = image([-24,8],[-8,8],ind2rgb(flipud(bg),map));
axis equal;
plot(X(1),Y(1),'ko');
plot(target(1),target(2),'kp');
xlim([-23.5,0]);
ylim([-6,6]);
% policy
subplot(2,1,2); hold on;
load([policyType '1CFD_policy-9/policyTest' num2str(round(mod(time(1)*20,90))/20,'%.2f') '.mat'],'actions');
colormap parula
theta_index = (Theta(1)+pi)/(pi/18) + 1;
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
    theta_index = (Theta(i)+pi)/(pi/18) + 1;
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
%% convert to mp4 and delete the original file
system(['ffmpeg -i ' policyType '_trajWithPolicy.avi' ...
    ' -c:v libx264 -profile:v high -crf 20 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" ' ...
    '-pix_fmt yuv420p ' policyType 'output.mp4 -y']);
system(['[ -f' policyType 'output.mp4 ] && rm ' policyType '_trajWithPolicy.avi']);
function [targetPos, initPos, trajX, trajY, trajTheta, time, reward] = getTrajectory(dir)
T = readlines([dir 'testTrajectory.txt']);
targetStrings = split(T(1));
targetX = str2double(targetStrings(3));
targetY = str2double(targetStrings(4));
initPosStrings = split(T(2));
initX = str2double(initPosStrings(4));
initY = str2double(initPosStrings(5));
initTheta = str2double(initPosStrings(6));
targetPos = [targetX, targetY];
initPos = [initX, initY, initTheta];
trajX = zeros(length(T)-3,1);
trajY = zeros(length(T)-3,1);
trajTheta = zeros(length(T)-3,1);
time = zeros(length(T)-3,1);

for i = 3:length(T)-1
    trajStrings = split(T(i));
    trajX(i-2) = str2double(trajStrings(2));
    trajY(i-2) = str2double(trajStrings(3));
    trajTheta(i-2) = str2double(trajStrings(4));
    time(i-2) = str2double(trajStrings(7));
end
rewardStrings = split(T(end));
reward = str2double(rewardStrings(2));
end