close all;
clear;
%%
figureDefaultSettings;
%%
n_angle = 36;
source = './';
%%

%% Old txt format

T = readlines([source 'success_region.txt']);
targetStrings = split(T(1));
targetX = str2double(targetStrings(3));
targetY = str2double(targetStrings(4));

initX = zeros(length(T)-2,1);
initY = zeros(length(T)-2,1);
initTheta = zeros(length(T)-2,1);
reward = zeros(length(T)-2,1);

for i = 1:length(T)-2
    resultStrings = split(T(i+1));
    initX(i) = str2double(resultStrings(2));
    initY(i) = str2double(resultStrings(3));
    temp = char(resultStrings(4));
    initTheta(i) = str2double(temp(1:end-1));
    reward(i) = str2double(resultStrings(end));
end
%% select colors

%% success rate
range = [0,2*pi];
% for k = 1:12
% ang_start = 3*k-3;
% ang_end = 3*k-1;
ang_start = 1;
ang_end = 36;
n_angle = 36;
name = 'geo1_long';
plot_successRate(source,name, n_angle, ang_start, ang_end);
name = 'egograd1_long';
plot_successRate(source,name, n_angle, ang_start, ang_end);
name = 'geo1';
plot_successRate(source,name, n_angle, ang_start, ang_end);
% % name = 'egograd5';
% % plot_successRate(source,name, n_angle, ang_start, ang_end);
name = 'egograd1';
plot_successRate(source,name, n_angle, ang_start, ang_end);
% 
name = 'far_egograd1_long';
plot_successRate(source,name, n_angle, ang_start, ang_end);
name = 'far_geo1_long';
plot_successRate(source,name, n_angle, ang_start, ang_end);
name = 'near_geo1';
plot_successRate(source,name, n_angle, ang_start, ang_end);
name = 'near_egograd1';
plot_successRate(source,name, n_angle, ang_start, ang_end);
% pause(2);
% end
%% time consumption
ang_start = 1;
ang_end = 36;
n_angle = 36;
name = 'egograd1_long';
plot_timeUsed(source,n_angle, name, ang_start,ang_end)
name = 'geo1_long';
plot_timeUsed(source,n_angle,name, ang_start,ang_end)
name = 'far_egograd1_long';
plot_timeUsed(source,n_angle, name, ang_start,ang_end)
name = 'far_geo1_long';
plot_timeUsed(source,n_angle,name, ang_start,ang_end)
name = 'near_egograd1';
plot_timeUsed(source,n_angle, name, ang_start,ang_end)
name = 'near_geo1';
plot_timeUsed(source,n_angle,name, ang_start,ang_end)


%%
load('success_region_trained_egosinglegrad2.mat')
figure("Position",[274 717 1461 443])
quiver(initX,initY,cos(initTheta),sin(initTheta),'o','AutoScaleFactor',0.6,'MarkerSize',3)
axis equal
hold on
plot(targetX,targetY,'o','MarkerSize',3,'LineWidth',0.5)
% plot([initX;targetX],[initY;targetY],'Color',[0.6,0.6,0.6],'linestyle','--','linewidth',0.5)
for i = 1:length(initX)
    if reward(i)<50
        plot([initX(i);targetX(i)],[initY(i);targetY(i)],'Color',[0.6,0.6,0.6],'linestyle','--','linewidth',0.5)
    end
%     p = plot(trajectory(i,1:totTime(i)+1,1),trajectory(i,1:totTime(i)+1,2),'Color',[0.5,0.5,0.5],'linestyle','--','linewidth',0.5);
end
% p.Color = [0.2,0.2,0.2];
% p.LineStyle = '-';
% p.LineWidth = 1;
xlim([-24,-6])
ylim([-6,6])
title(['Success rate: ' num2str(sum(reward>50)/length(reward))])
axis off
%% plot selected test points for original task

% load('success_region_trained_geo1.mat')
% selected = randperm(length(initX),20);
selected = find(reward<50);
figure("Position",[274 717 1461 443])
subplot(1,2,1),quiver(initX,initY,cos(initTheta),sin(initTheta),'o','AutoScaleFactor',0.6,'MarkerSize',3)
axis equal
hold on
plot(targetX,targetY,'o','MarkerSize',3,'LineWidth',0.5)
% plot([initX;targetX],[initY;targetY],'Color',[0.6,0.6,0.6],'linestyle','--','linewidth',0.5)
for i = 1:length(selected)
    k = selected(i);
    p = plot(trajectory(k,1:totTime(k)+1,1),trajectory(k,1:totTime(k)+1,2),'Color',[0.5,0.5,0.5],'linestyle','--','linewidth',0.5);
end
p.Color = [0.2,0.2,0.2];
p.LineStyle = '-';
p.LineWidth = 1;
xlim([-24,-6])
ylim([-6,6])
title(['Success rate: ' num2str(sum(reward>50)/length(reward))])
axis off
load('success_region_trained_egograd1.mat')
selected = find(reward<50);
subplot(1,2,2),quiver(initX,initY,cos(initTheta),sin(initTheta),'o','AutoScaleFactor',0.6,'MarkerSize',3)
axis equal
hold on
plot(targetX,targetY,'o','MarkerSize',3,'LineWidth',0.5)
% plot([initX;targetX],[initY;targetY],'Color',[0.6,0.6,0.6],'linestyle','--','linewidth',0.5)
for i = 1:length(selected)
    k = selected(i);
    p = plot(trajectory(k,1:totTime(k)+1,1),trajectory(k,1:totTime(k)+1,2),'Color',[0.5,0.5,0.5],'linestyle','--','linewidth',0.5);
end
p.Color = [0.2,0.2,0.2];
p.LineStyle = '-';
p.LineWidth = 1;
xlim([-24,-6])
ylim([-6,6])
title(['Success rate: ' num2str(sum(reward>50)/length(reward))])
axis off
%%
function plot_trained_region(source, name, n_angle)
load([source 'success_region_trained_' name '.mat'])
sr = zeros(1,length(initX));

list(list<1) = list(list<1) + 36;
success = reward(i-1+list)>50;
sr((i-1)/n_angle+1) = sum(success);

figure('Position',[960 848 640 284]);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[8,-8],ind2rgb(bg,map));
plot(targetX,targetY,'*',Color=[50/255,100/255,50/255]);
title(['initial orientation: ' num2str(2/n_angle*(ang_start-1),3) '$\pi$ - ' num2str(2/n_angle*(ang_end-1),3) '$\pi$'])
% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
colormap parula
imagesc([min(initX),max(initX)],[min(initY),max(initY)],reshape(sr,length(unique(initY)),[]),'AlphaData',0.8);
axis equal;
colorbar("Ticks",[],'Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
end
%%
function plot_successRate(souce, name, n_angle,ang_start,ang_end)
load([souce 'success_region_' name '.mat'])
targetX = target(1);
targetY = target(2);
sr = zeros(1,length(initX)/n_angle);
range = ang_end - ang_start + 1;

for i = 1:n_angle:length(initY)
    %     success = reward(i:i+n_angle-1)>50;
    %     sr(i) = sum(success)/n_angle;
    list = ang_start:ang_end;
    list(list<1) = list(list<1) + n_angle;
    success = reward(i-1+list)>50;
    sr((i-1)/n_angle+1) = sum(success)/range;
end

figure('Position',[960 848 640 284]);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[8,-8],ind2rgb(bg,map));
plot(targetX,targetY,'p',Color=[50/255,100/255,50/255]);
success = reward>50;
sr_overall = mean(success);
is_trained = ((initX + 12).^2 + (initY + 2.15).^2 <= 4);
sr_trained = mean(success(is_trained));
sr_untrained = mean(success(~is_trained));
title([replace(name,'_',' ') ', untrained:' num2str(100*sr_untrained,'%.2f') '\%, trained:' num2str(100*sr_trained,'%0.2f') '\%']);
% title(['initial orientation: ' num2str(2/n_angle*(ang_start-1)) '$\pi$ - ' num2str(2/n_angle*(ang_end-1)) '$\pi$'])
% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
cmap = cbrewer('seq','YlGn',400,'linear');
colormap(cmap)

% imagesc([min(initX),max(initX)],[min(initY),max(initY)],reshape(sr,length(unique(initY)),[]),'AlphaData',0.8);
x = initX(1:n_angle:end);y = initY(1:n_angle:end);
s = scatter(x,y,24,sr,'filled');
% s.AlphaData = sr;
% s.MarkerFaceAlpha = 'flat';
axis equal;
colorbar('Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
exportgraphics(gcf,['./savedFigs/' name '_successrate.eps'])
end
%%
function plot_timeUsed(source,n_angle,name, ang_start,ang_end)
load([source 'success_region_' name '.mat'])
targetX = target(1); targetY = target(2);
success = reward > 50;
totTime(~success) = max(totTime);
tc = zeros(1,length(initX)/n_angle);

range = ang_end - ang_start + 1;
for i = 1:n_angle:length(initY)
    %     success = reward(i:i+n_angle-1)>50;
    %     sr(i) = sum(success)/n_angle;
    list = ang_start:ang_end;
    list(list<1) = list(list<1) + n_angle;
    t_list = totTime(list+i-1);
    tc((i-1)/n_angle+1) = mean(t_list(t_list<max(totTime)));
end
figure('Position',[960 848 640 284]);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[8,-8],ind2rgb(bg,map));
plot(targetX,targetY,'p',Color=[50/255,100/255,50/255]);
title([replace(name,'_',' ') ', ' num2str(sum(success)/length(initX))]);
% title(['initial orientation: ' num2str(1/18*(ang_start-1)) '$\pi$ - ' num2str(1/18*(ang_end-1)) '$\pi$'])
% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(cmap)
% imagesc([min(initX),max(initX)],[min(initY),max(initY)],reshape(tc,length(unique(initY)),[]),'AlphaData',0.8);
x = initX(1:n_angle:end);y = initY(1:n_angle:end);
s = scatter(x,y,24,tc,'filled');
axis equal;
% caxis([0,200*ceil(max(tc)/200)]);
caxis([0,800]);
colorbar('Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
exportgraphics(gcf,['./savedFigs/' name '_timeconsumption.eps'])
end
