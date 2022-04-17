% close all;
clear;
%%
figureDefaultSettings;
%%
n_angle = 36;
dir = './';
%%

%% Old txt format

% T = readlines([dir 'success_region.txt']);
% targetStrings = split(T(1));
% targetX = str2double(targetStrings(3));
% targetY = str2double(targetStrings(4));
% 
% initX = zeros(length(T)-2,1);
% initY = zeros(length(T)-2,1);
% initTheta = zeros(length(T)-2,1);
% reward = zeros(length(T)-2,1);
% 
% for i = 1:length(T)-2
%     resultStrings = split(T(i+1));
%     initX(i) = str2double(resultStrings(2));
%     initY(i) = str2double(resultStrings(3));
%     temp = char(resultStrings(4));
%     initTheta(i) = str2double(temp(1:end-1));
%     reward(i) = str2double(resultStrings(end));
% end

%% success rate
range = [0,2*pi];
% for k = 1:12
% ang_start = 3*k-3;
% ang_end = 3*k-1;
ang_start = 1;
ang_end = 36;
plot_successRate(n_angle, initX, initY, reward, target,ang_start,ang_end);
% pause(2);
% end
%% time consumption
ang_start = 1;
ang_end = 36;
name = 'egograd1';
plot_timeUsed(dir,n_angle, name, ang_start,ang_end)
name = 'egograd5';
plot_timeUsed(dir,n_angle,name, ang_start,ang_end)
name = 'lab1';
plot_timeUsed(dir,n_angle,name, ang_start,ang_end)
name = 'lab4';
plot_timeUsed(dir,n_angle,name, ang_start,ang_end)

%%
function plot_successRate(n_angle, initX, initY, reward, target,ang_start,ang_end)
targetX = target(1);
targetY = target(2);
sr = zeros(1,length(initX)/n_angle);
range = ang_end - ang_start + 1;
for i = 1:n_angle:length(initY)
%     success = reward(i:i+n_angle-1)>50;
%     sr(i) = sum(success)/n_angle;
    list = ang_start:ang_end;
    list(list<1) = list(list<1) + 36;
    success = reward(i-1+list)>50;
    sr((i-1)/n_angle+1) = sum(success)/range;
end

figure('Position',[960 848 640 284]);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[-8,8],ind2rgb(bg,map));
plot(targetX,targetY,'*',Color=[50/255,100/255,50/255]);
title(['initial orientation: ' num2str(1/18*(ang_start-1)) '$\pi$ - ' num2str(1/18*(ang_end-1)) '$\pi$'])
% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
colormap parula
imagesc([min(initX),max(initX)],[min(initY),max(initY)],reshape(sr,9,[]),'AlphaData',0.8);
axis equal;
colorbar("Ticks",[],'Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
end
function plot_timeUsed(dir,n_angle,name, ang_start,ang_end)
load([dir 'success_region_' name '.mat'])
targetX = target(1); targetY = target(2);
success = reward > 50;
totTime(~success) = max(totTime);
tc = zeros(1,length(initX)/n_angle);

range = ang_end - ang_start + 1;
for i = 1:n_angle:length(initY)
%     success = reward(i:i+n_angle-1)>50;
%     sr(i) = sum(success)/n_angle;
    list = ang_start:ang_end;
    list(list<1) = list(list<1) + 36;
    t_list = totTime(list+i-1);
    tc((i-1)/n_angle+1) = mean(t_list(t_list<max(totTime)));
end
figure('Position',[960 848 640 284]);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[8,-8],ind2rgb(bg,map));
plot(targetX,targetY,'*',Color=[50/255,100/255,50/255]);
title([name ',' num2str(sum(success)/length(initX))]);
% title(['initial orientation: ' num2str(1/18*(ang_start-1)) '$\pi$ - ' num2str(1/18*(ang_end-1)) '$\pi$'])
% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
colormap jet
imagesc([min(initX),max(initX)],[min(initY),max(initY)],reshape(tc,9,[]),'AlphaData',0.8);
axis equal;
caxis([0,600]);
colorbar('Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
end
