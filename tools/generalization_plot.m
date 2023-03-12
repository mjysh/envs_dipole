close all;
clear;
%%
figureDefaultSettings;
%%
n_angle = 36;
ang_start = 1;
ang_end = 36;
source = '/home/yusheng/smarties/apps/dipole_adapt/paper_new/';
%% select colors

%% success rate
range = [0,2*pi];
% for k = 1:12
% ang_start = 3*k-3;
% ang_end = 3*k-1;


targetX = -12;
targetY = 2.15;
policy_name = 'geo3';
[success_geo, is_trained_geo, totTime_geo] = test_varySwimmer_plot(source, policy_name, n_angle,ang_start,ang_end,targetX,targetY);
%%
policy_name = 'egoLRGrad1';
% [success_ego, is_trained_ego, totTime_ego] = test_varySwimmer_plot(source, policy_name, n_angle,ang_start,ang_end,targetX,targetY);
swimmerX = -12;
swimmerY = -2.15;
env = 'ego2sensorLRGradCFD';
[success_ego, is_trained_ego, totTime_ego] = test_varyTarget_plot(source, env,policy_name, n_angle,ang_start,ang_end, swimmerX, swimmerY);
%%
policy_name = 'geo13';
% [success_ego, is_trained_ego, totTime_ego] = test_varySwimmer_plot(source, policy_name, n_angle,ang_start,ang_end,targetX,targetY);
swimmerX = -12;
swimmerY = -2.15;
env = 'geo1sensorCFD';
[success_ego, is_trained_ego, totTime_ego] = test_varyTarget_plot(source, env,policy_name, n_angle,ang_start,ang_end, swimmerX, swimmerY);
%%
targetX = -12;
targetY = 2.15;
% policy_name = 'georeduced1';
% env = 'georeduced_widebound';
policy_name = 'egoLRGrad1';
env = 'ego2sensorLRGradCFD';
% policy_name = 'egoDirLRGradreduced3';
% env = 'egoDirLRGradreduced_widebound';
[success_geo, is_trained_geo, totTime_geo] = test_varySwimmer_plot(source, env, policy_name, n_angle,ang_start,ang_end,targetX,targetY);
%%
rewardX = -8;
policy_name = 'sourceseeking_reduced1';
env = 'egoLRGradSourceseekingreduced';
env = 'egoLRGradSourceseekingreduced';
[success_ego, is_trained_ego, totTime_ego] = sourceseeking_result_plot(source, policy_name,env,n_angle,ang_start,ang_end,rewardX);
%%
shared_geo = totTime_geo(success_ego&success_geo);
shared_ego = totTime_ego(success_ego&success_geo);
distinct_geo = totTime_geo(~success_ego&success_geo);
distinct_ego = totTime_ego(success_ego&~success_geo);
% trainedTime_geo = totTime_geo(success_ego&success_geo&is_trained_geo);
% trainedTime_ego = totTime_ego(success_ego&success_geo&is_trained_geo);
% untrainedTime_geo = totTime_geo(success_ego&success_geo&~is_trained_geo);
% untrainedTime_ego = totTime_ego(success_ego&success_geo&~is_trained_geo);
%%
edges = 0:50:1000;
N_shared_geo = histcounts(shared_geo,edges);
N_shared_ego = histcounts(shared_ego,edges);
N_distinct_geo = histcounts(distinct_geo,edges);
N_distinct_ego = histcounts(distinct_ego,edges);
% figure,bar(edges(1:end-1),[N_shared_geo' N_shared_ego' N_distinct_geo' N_distinct_ego'])
figure('Position',[420 604 1731 230]),bar(edges(1:end-1),[N_shared_geo' N_distinct_geo'],'stacked');
ylim([0,3000]);
figure('Position',[420 604 1731 230]),bar(edges(1:end-1),[N_shared_ego' N_distinct_ego'],'stacked');
ylim([0,3000]);
%%
c=redblue(100)
figure;
colormap(c);
colorbar
%%
function [success, is_trained, totTime] = test_varySwimmer_plot(source, env,policy_name, n_angle,ang_start,ang_end, targetX, targetY)
 load([source policy_name '/success_region' num2str(targetX) '_' num2str(targetY) '_' env '.mat'], ...
    'reward','totTime','initX','initY','initTheta','target')
assert(targetX == target(1));
assert(targetY == target(2));

selected = initX > -23.2;
initX = initX(selected);
initY = initY(selected);
initTheta = initTheta(selected);
reward = reward(selected);
totTime = totTime(selected);


sr = zeros(1,length(initX)/n_angle);
tc = zeros(1,length(initX)/n_angle);

success = reward>50;
totTime(~success) = max(totTime);

range = ang_end - ang_start + 1;
for i = 1:n_angle:length(initY)
    %     success = reward(i:i+n_angle-1)>50;
    %     sr(i) = sum(success)/n_angle;
    list = ang_start:ang_end;
    list(list<1) = list(list<1) + n_angle;
    success_loc = reward(i-1+list)>50;
    sr((i-1)/n_angle+1) = sum(success_loc)/range;

    t_list = totTime(list+i-1);
    tc((i-1)/n_angle+1) = mean(t_list(t_list<max(totTime)));
end


figure('Position',[960 848 640 284]);
if contains(env,'reduced')
    lam = 4;
    A = 0.3;
    vortexUpX = mod(24,lam)-24:lam:0;
    vortex_up = plot(vortexUpX,A*ones(size(vortexUpX)),'r.','MarkerSize',24); hold on
    vortexDownX = mod(24+lam/2,lam)-24:lam:0;
    vortex_down = plot(vortexDownX,-A*ones(size(vortexDownX)),'b.','MarkerSize',24);
else
    [bg,map] = imread("movie2000.png","png");hold on;
    image([-24,8],[8,-8],ind2rgb(bg,map));
end
plot(targetX,targetY,'p',Color=[50/255,100/255,50/255]);
sr_overall = mean(success);
is_trained = ((initX + 12).^2 + (initY + 2.15).^2 <= 4);
sr_trained = mean(success(is_trained));
sr_untrained = mean(success(~is_trained));
title([replace(policy_name,'_',' ') ', untrained:' num2str(100*sr_untrained,'%.2f') '\%, trained:' num2str(100*sr_trained,'%0.2f') '\%']);
cmap = cbrewer('seq','YlGn',400,'linear');
colormap(cmap)

x = initX(1:n_angle:end);y = initY(1:n_angle:end);
s = scatter(x(sr>0),y(sr>0),24,sr(sr>0),'filled');
axis equal;
colorbar('Location','westoutside')
xlim([min(min(initX),-23.5),max(max(initX),0)]);
ylim([min(min(initY),-6),max(max(initY),6)]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
exportgraphics(gcf,['./savedFigs/' policy_name '_' num2str(targetX) '_' num2str(targetY) '_successrate.eps'])

figure('Position',[960 848 640 284]);
if contains(env,'reduced')
    lam = 4;
    A = 0.3;
    vortexUpX = mod(24,lam)-24:lam:0;
    vortex_up = plot(vortexUpX,A*ones(size(vortexUpX)),'r.','MarkerSize',24); hold on
    vortexDownX = mod(24+lam/2,lam)-24:lam:0;
    vortex_down = plot(vortexDownX,-A*ones(size(vortexDownX)),'b.','MarkerSize',24);
else
    [bg,map] = imread("movie2000.png","png");hold on;
    image([-24,8],[8,-8],ind2rgb(bg,map));
end
plot(targetX,targetY,'p',Color=[50/255,100/255,50/255]);
title([replace(policy_name,'_',' ') ', ' num2str(sr_overall)]);
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(flipud(cmap))
x = initX(1:n_angle:end);y = initY(1:n_angle:end);
s = scatter(x,y,24,tc,'filled');
axis equal;
caxis([0,1000]);
colorbar('Location','westoutside')
xlim([min(min(initX),-23.5),max(max(initX),0)]);
ylim([min(min(initY),-6),max(max(initY),6)]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
exportgraphics(gcf,['./savedFigs/' policy_name '_' num2str(targetX) '_' num2str(targetY) '_timeconsumption.eps'])

end
%%
function [success, is_trained, totTime] = test_varyTarget_plot(source, env,policy_name, n_angle,ang_start,ang_end, swimmerX, swimmerY)
 load([source policy_name '/success_region_varytarget_' num2str(swimmerX) '_' num2str(swimmerY) '_' env '.mat'], ...
    'reward','totTime','initX','initY','initTheta','targetX','targetY')
assert(swimmerX == initX);
assert(swimmerY == initY);

% initX = initX(selected);
% initY = initY(selected);
% initTheta = initTheta(selected);
% reward = reward(selected);
% totTime = totTime(selected);

testN = length(targetX);

sr = zeros(1,testN/n_angle);
tc = zeros(1,testN/n_angle);

success = reward>50;
totTime(~success) = max(totTime);

range = ang_end - ang_start + 1;
for i = 1:n_angle:testN
    %     success = reward(i:i+n_angle-1)>50;
    %     sr(i) = sum(success)/n_angle;
    list = ang_start:ang_end;
    list(list<1) = list(list<1) + n_angle;
    success_loc = reward(i-1+list)>50;
    sr((i-1)/n_angle+1) = sum(success_loc)/range;

    t_list = totTime(list+i-1);
    tc((i-1)/n_angle+1) = mean(t_list(t_list<max(totTime)));
end


figure('Position',[960 848 640 284]);
if contains(env,'reduced')
    lam = 4;
    A = 0.3;
    vortexUpX = mod(24,lam)-24:lam:0;
    vortex_up = plot(vortexUpX,A*ones(size(vortexUpX)),'r.','MarkerSize',24); hold on
    vortexDownX = mod(24+lam/2,lam)-24:lam:0;
    vortex_down = plot(vortexDownX,-A*ones(size(vortexDownX)),'b.','MarkerSize',24);
else
    [bg,map] = imread("movie2000.png","png");hold on;
    image([-24,8],[8,-8],ind2rgb(bg,map));
end
plot(swimmerX,swimmerY,'p',Color=[50/255,100/255,50/255]);
sr_overall = mean(success);
is_trained = ((targetX + 12).^2 + (targetY - 2.15).^2 <= 4);
sr_trained = mean(success(is_trained));
sr_untrained = mean(success(~is_trained));
title([replace(policy_name,'_',' ') ', untrained:' num2str(100*sr_untrained,'%.2f') '\%, trained:' num2str(100*sr_trained,'%0.2f') '\%']);
cmap = cbrewer('seq','YlGn',400,'linear');
colormap(cmap)

x = targetX(1:n_angle:end);y = targetY(1:n_angle:end);
s = scatter(x(sr>0),y(sr>0),24,sr(sr>0),'filled');
axis equal;
colorbar('Location','westoutside')
xlim([min(min(targetX),-23.5),max(max(targetX),0)]);
ylim([min(min(targetY),-6),max(max(targetY),6)]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
exportgraphics(gcf,['./savedFigs/' policy_name 'varytarget_' num2str(swimmerX) '_' num2str(swimmerY) '_successrate.eps'])

figure('Position',[960 848 640 284]);
if contains(env,'reduced')
    lam = 4;
    A = 0.3;
    vortexUpX = mod(24,lam)-24:lam:0;
    vortex_up = plot(vortexUpX,A*ones(size(vortexUpX)),'r.','MarkerSize',24); hold on
    vortexDownX = mod(24+lam/2,lam)-24:lam:0;
    vortex_down = plot(vortexDownX,-A*ones(size(vortexDownX)),'b.','MarkerSize',24);
else
    [bg,map] = imread("movie2000.png","png");hold on;
    image([-24,8],[8,-8],ind2rgb(bg,map));
end
plot(swimmerX,swimmerY,'p',Color=[50/255,100/255,50/255]);
title([replace(policy_name,'_',' ') ', ' num2str(sr_overall)]);
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(flipud(cmap))
x = targetX(1:n_angle:end);y = targetY(1:n_angle:end);
s = scatter(x,y,24,tc,'filled');
axis equal;
caxis([0,1000]);
colorbar('Location','westoutside')
xlim([min(min(targetX),-23.5),max(max(targetX),0)]);
ylim([min(min(targetY),-6),max(max(targetY),6)]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
exportgraphics(gcf,['./savedFigs/' policy_name 'varytarget_' num2str(swimmerX) '_' num2str(swimmerY) '_timeconsumption.eps'])

end
%%
function [success, is_trained, totTime] = sourceseeking_result_plot(source, policy_name, env,n_angle,ang_start,ang_end,rewardX)
load([source policy_name '/success_region_sourceseeking_' env num2str(rewardX) '.mat'], ...
    'reward','totTime','initX','initY','initTheta','threshold')
% assert(rewardX == threshold)
selected = initX > -23.2;
initX = initX(selected);
initY = initY(selected);
initTheta = initTheta(selected);
reward = reward(selected);
totTime = totTime(selected);


sr = zeros(1,length(initX)/n_angle);
tc = zeros(1,length(initX)/n_angle);

success = reward>50;
totTime(~success) = max(totTime);

range = ang_end - ang_start + 1;
for i = 1:n_angle:length(initY)
    %     success = reward(i:i+n_angle-1)>50;
    %     sr(i) = sum(success)/n_angle;
    list = ang_start:ang_end;
    list(list<1) = list(list<1) + n_angle;
    success_loc = reward(i-1+list)>50;
    sr((i-1)/n_angle+1) = sum(success_loc)/range;

    t_list = totTime(list+i-1);
    tc((i-1)/n_angle+1) = mean(t_list(t_list<max(totTime)));
end


figure('Position',[960 848 640 284]);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[8,-8],ind2rgb(bg,map));

sr_overall = mean(success);
is_trained = (initX>=-12) & (initX<=-8) & (initY >= -2) & (initY<=2);
title([replace(policy_name,'_',' ') ]);
cmap = cbrewer('seq','YlGn',400,'linear');
colormap(cmap)

x = initX(1:n_angle:end);y = initY(1:n_angle:end);
s = scatter(x,y,24,sr,'filled');
axis equal;
colorbar('Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
xx = 0:0.01:1;
plot(-12+4*xx,2*ones(size(xx)),'k');
plot(-12+4*xx,-2*ones(size(xx)),'k');
plot(-8*ones(size(xx)),-2+4*xx,'k');
plot(-12*ones(size(xx)),-2+4*xx,'k');

exportgraphics(gcf,['./savedFigs/' policy_name '_' num2str(rewardX) '_successrate.eps'])

figure('Position',[960 848 640 284]);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[8,-8],ind2rgb(bg,map));
title([replace(policy_name,'_',' ') ', ' num2str(sr_overall)]);
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(flipud(cmap))
x = initX(1:n_angle:end);y = initY(1:n_angle:end);
s = scatter(x,y,24,tc,'filled');
axis equal;
caxis([0,1000]);
colorbar('Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
xx = 0:0.01:1;
plot(-12+4*xx,2*ones(size(xx)),'k');
plot(-12+4*xx,-2*ones(size(xx)),'k');
plot(-8*ones(size(xx)),-2+4*xx,'k');
plot(-12*ones(size(xx)),-2+4*xx,'k');
exportgraphics(gcf,['./savedFigs/' policy_name '_' num2str(rewardX) '_timeconsumption.eps'])
end

%%
function [totTime, is_trained] = plot_timeUsed(source, policy_name, n_angle,ang_start,ang_end, targetX, targetY)
load([source policy_name '/success_region' num2str(targetX) '_' num2str(targetY) '.mat'])
assert(targetX == target(1));
assert(targetY == target(2));
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
title([replace(policy_name,'_',' ') ', ' num2str(sum(success)/length(initX))]);
% title(['initial orientation: ' num2str(1/18*(ang_start-1)) '$\pi$ - ' num2str(1/18*(ang_end-1)) '$\pi$'])
% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(flipud(cmap))
% imagesc([min(initX),max(initX)],[min(initY),max(initY)],reshape(tc,length(unique(initY)),[]),'AlphaData',0.8);
x = initX(1:n_angle:end);y = initY(1:n_angle:end);
s = scatter(x,y,24,tc,'filled');
axis equal;
% caxis([0,200*ceil(max(tc)/200)]);
caxis([0,1000]);
colorbar('Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
exportgraphics(gcf,['./savedFigs/' policy_name '_' num2str(targetX) '_' num2str(targetY) '_timeconsumption.eps'])
end

function c = redblue(m)
%REDBLUE    Shades of red and blue color map
%   REDBLUE(M), is an M-by-3 matrix that defines a colormap.
%   The colors begin with bright blue, range through shades of
%   blue to white, and then through shades of red to bright red.
%   REDBLUE, by itself, is the same length as the current figure's
%   colormap. If no figure exists, MATLAB creates one.
%
%   For example, to reset the colormap of the current figure:
%
%             colormap(redblue)
%
%   See also HSV, GRAY, HOT, BONE, COPPER, PINK, FLAG, 
%   COLORMAP, RGBPLOT.
%   Adam Auton, 9th October 2009
if nargin < 1, m = size(get(gcf,'colormap'),1); end
if (mod(m,2) == 0)
    % From [0 0 1] to [1 1 1], then [1 1 1] to [1 0 0];
    m1 = m*0.5;
    r = (0:m1-1)'/max(m1-1,1);
    g = r;
    r = [r; ones(m1,1)];
    g = [g; flipud(g)];
    b = flipud(r);
else
    % From [0 0 1] to [1 1 1] to [1 0 0];
    m1 = floor(m*0.5);
    r = (0:m1-1)'/max(m1,1);
    g = r;
    r = [r; ones(m1+1,1)];
    g = [g; 1; flipud(g)];
    b = flipud(r);
end
c = [r g b]; 
end