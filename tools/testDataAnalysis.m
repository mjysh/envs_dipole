close all;clear;
figureDefaultSettings;
%%
root_dir = '/home/yusheng/smarties/apps/dipole_adapt/paper_new/';
load([root_dir 'georeduced1/grading_results_geo1sensorreduced.mat']);
%% 
policy_name = 'egoLRGrad1';
load([root_dir policy_name '/grading_results_ego2sensorLRGradCFD.mat']);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[8,-8],ind2rgb(bg,map));
axis equal;
N = 222;
tx = trajX(:,N);
ty = trajY(:,N);
plot(tx(abs(tx)>0),ty(abs(tx)>0),'k');
the = 0:pi/200:pi*2;
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
plot(-12+2*cos(the),2.15+2*sin(the),'k');
% colorbar('Location','westoutside')
%% grading results of egoLRGrad1
% policy_name = 'egoLRGrad1';
% load([root_dir policy_name '/grading_results_ego2sensorLRGradCFD.mat']);
policy_name = 'egoLRGradreduced1';
env = 'egoLRGradreduced';
% load([root_dir policy_name '/grading_results_' env '.mat']);
% policy_name = 'georeduced1';
% env = 'geo1sensorreduced';
load([root_dir policy_name '/grading_results_' env '.mat']);
%%
% load([root_dir 'egoLRGrad1/grading_results_egoLRGradreduced.mat']);
success = reward>50;
figure('position',[297 850 630 472]);
sgtitle('egoLRGrad1')
cmap = cbrewer('seq','YlGn',400,'linear');
colormap(cmap)
xedges = -14:0.4:-10;
yedges = 0.15:0.4:4.15;
[X,Y] = meshgrid((xedges(1:end-1)+xedges(2:end))/2,(yedges(1:end-1)+yedges(2:end))/2);
[hc_end,xedges,yedges,xbin,ybin] = histcounts2(target(:,1),target(:,2),xedges,yedges);
hc_end_success = histcounts2(target(success,1),target(success,2),xedges,yedges);
tc = zeros(size(X));
for i = 1:length(totTime)
    if success(i)
        tc(xbin(i),ybin(i)) = tc(xbin(i),ybin(i)) + totTime(i);
    end
end
tc = tc'./hc_end_success';
sr = hc_end_success'./hc_end';
subplot(2,2,1),scatter(X(:),Y(:),50,sr(:),'filled');
% hc_end = [hc_end zeros(size(hc_end,1),1)];hc_end = [hc_end;zeros(1,size(hc_end,2))];
% hc_end_success = [hc_end_success zeros(size(hc_end_success,1),1)];
% hc_end_success = [hc_end_success;zeros(1,size(hc_end_success,2))];
% subplot(2,2,1),h_end = pcolor(X',Y',hc_end_success./hc_end);
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([0.15,4.15]);
yticks(0.15:2:4.15)
colorbar;
clim([0,1])

ax2=subplot(2,2,2);
scatter(X(:),Y(:),50,tc(:),'filled');
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(ax2,flipud(cmap))
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([0.15,4.15]);
yticks(0.15:2:4.15)
colorbar;
clim([0,800])


xedges = -14:0.4:-10;
yedges = -4.15:0.4:-0.15;
[X,Y] = meshgrid((xedges(1:end-1)+xedges(2:end))/2,(yedges(1:end-1)+yedges(2:end))/2);
[hc_start,xedges,yedges,xbin,ybin] = histcounts2(trajX(1,:),trajY(1,:),xedges,yedges);
hc_start_success = histcounts2(trajX(1,success),trajY(1,success),xedges,yedges);
tc = zeros(size(X));
for i = 1:length(totTime)
    if success(i)
        tc(xbin(i),ybin(i)) = tc(xbin(i),ybin(i)) + totTime(i);
    end
end
tc = tc'./hc_start_success';
sr = hc_start_success'./hc_start';
subplot(2,2,3),scatter(X(:),Y(:),50,sr(:),'filled');
% hc_start = [hc_start zeros(size(hc_start,1),1)];hc_start = [hc_start;zeros(1,size(hc_start,2))];
% hc_start_success = [hc_start_success zeros(size(hc_start_success,1),1)];
% hc_start_success = [hc_start_success;zeros(1,size(hc_start_success,2))];
% subplot(2,2,3),h_start = pcolor(X',Y',hc_start_success./hc_start);
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([-4.15,-0.15]);
yticks(-4.15:2:-0.15)
colorbar;
clim([0,1])

ax4=subplot(2,2,4);
scatter(X(:),Y(:),50,tc(:),'filled');
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(ax4,flipud(cmap))
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([-4.15,-0.15]);
yticks(-4.15:2:-0.15)
colorbar;
clim([0,800])
sgtitle(policy_name);
exportgraphics(gcf,['./savedFigs/grading_analysis_' policy_name env '_ratetime.eps'],'ContentType','vector')
%%
figure('Position',[705 13 886 1309]);
obs_x = squeeze(observation(1,:,:)); obs_y = squeeze(observation(2,:,:));
obs_x = obs_x(:); obs_y = obs_y(:);
obs_x = obs_x(abs(obs_x)>0);obs_y = obs_y(abs(obs_y)>0);
xbin = round((max(obs_x)-min(obs_x))/0.17);
ybin = round((max(obs_y)-min(obs_y))/0.17);
[hc_obs,xedges,yedges] = histcounts2(obs_x,obs_y,[xbin, ybin],'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges);
X = X'; Y= Y';
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax = subplot(4,1,1); h_obs = pcolor(X,Y,hc_obs);
hold on, plot(X(hc_obs==0),Y(hc_obs==0),'k.','MarkerSize',2)
axis equal;
xlim([min(xedges),max(xedges)]);
ylim([min(yedges),max(yedges)]);
xlabel('$\Delta x$');ylabel('$\Delta y$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
% map = flip(hot);
map = ones(400,3);
map(:,1) = linspace(1,0.84,400);
map(:,2) = linspace(1,0.16,400);
map(:,3) = linspace(1,0.16,400);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

obs_u = squeeze(observation(3,:,:)); obs_v = squeeze(observation(4,:,:));
obs_u = obs_u(:); obs_v = obs_v(:);
obs_u = obs_u(abs(obs_u)>0);obs_v = obs_v(abs(obs_v)>0);
ubin = round((max(obs_u)-min(obs_u))/0.03);
vbin = round((max(obs_v)-min(obs_v))/0.03);
[hc_obs,xedges,yedges] = histcounts2(obs_u,obs_v,[ubin,vbin],'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges); X = X'; Y= Y';
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax=subplot(4,1,2); h_obs = pcolor(X,Y,hc_obs);
hold on, plot(X(hc_obs==0),Y(hc_obs==0),'k.','MarkerSize',2)
axis equal;
xlim([min(xedges),max(xedges)]);
ylim([min(yedges),max(yedges)]);
xlabel('$u$');ylabel('$v$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
% map = flip(gray);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

obs_dudy = squeeze(observation(5,:,:)); obs_dvdy = squeeze(observation(6,:,:));
obs_dudy = obs_dudy(:); obs_dvdy = obs_dvdy(:);
obs_dudy = obs_dudy(abs(obs_dudy)>0);obs_dvdy = obs_dvdy(abs(obs_dvdy)>0);
limit_dudy = 2;
limit_dvdy = 0.9;
flag = abs(obs_dudy)<limit_dudy & abs(obs_dvdy)<limit_dvdy;
obs_dudy = obs_dudy(flag);obs_dvdy = obs_dvdy(flag);
% dudybin = round((max(obs_dudy)-min(obs_dudy))/0.02);
% dvdybin = round((max(obs_dvdy)-min(obs_dvdy))/0.02);
% dudybin = round(2*limit_dudy/0.02);
% dvdybin = round(2*limit_dvdy/0.02);
xedges = -limit_dudy:0.02:limit_dudy;
yedges = -limit_dvdy:0.02:limit_dvdy;
[hc_obs,xedges,yedges] = histcounts2(obs_dudy,obs_dvdy,xedges,yedges,'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges); X = X'; Y= Y';
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax=subplot(4,1,3);h_obs = pcolor(X,Y,hc_obs);
% Z = ones(size(X));
hold on, plot(X(hc_obs==0),Y(hc_obs==0),'k.','MarkerSize',2)
axis equal;
% xlim([min(xedges),max(xedges)]);
% ylim([min(yedges),max(yedges)]);
xlim([-limit_dudy,limit_dudy]);
ylim([-limit_dvdy,limit_dvdy]);
xlabel('$\partial u/\partial y$');ylabel('$\partial v/\partial y$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
% map = flip(gray);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

actions = 4*action(:);
% subplot(2,2,4),h_act = histfit(action(abs(action)>0));
subplot(4,1,4),h_act = histogram(actions(abs(actions)>0),'Normalization','pdf');
xlabel('$\dot\theta$');ylabel('PDF');
xlim([-4,4]);
ylim([0, 1.5]);
h_act(1).EdgeAlpha = 0;
exportgraphics(gcf,['./savedFigs/grading_analysis_' policy_name env '_obsact.eps'],'ContentType','vector')
%% grading results of geo13
policy_name = 'geo13';
load([root_dir policy_name '/grading_results_geo1sensorCFD.mat']);
%%
success = reward>50;
figure('position',[297 850 630 472]);
sgtitle('egoLRGrad1')
cmap = cbrewer('seq','YlGn',400,'linear');
colormap(cmap)
xedges = -14:0.4:-10;
yedges = 0.15:0.4:4.15;
[X,Y] = meshgrid((xedges(1:end-1)+xedges(2:end))/2,(yedges(1:end-1)+yedges(2:end))/2);
[hc_end,xedges,yedges,xbin,ybin] = histcounts2(target(:,1),target(:,2),xedges,yedges);
hc_end_success = histcounts2(target(success,1),target(success,2),xedges,yedges);
tc = zeros(size(X));
for i = 1:length(totTime)
    if success(i)
        tc(xbin(i),ybin(i)) = tc(xbin(i),ybin(i)) + totTime(i);
    end
end
tc = tc'./hc_end_success';
sr = hc_end_success'./hc_end';
subplot(2,2,1),scatter(X(:),Y(:),50,sr(:),'filled');
% hc_end = [hc_end zeros(size(hc_end,1),1)];hc_end = [hc_end;zeros(1,size(hc_end,2))];
% hc_end_success = [hc_end_success zeros(size(hc_end_success,1),1)];
% hc_end_success = [hc_end_success;zeros(1,size(hc_end_success,2))];
% subplot(2,2,1),h_end = pcolor(X',Y',hc_end_success./hc_end);
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([0.15,4.15]);
yticks(0.15:2:4.15)
colorbar;
clim([0,1])

ax2=subplot(2,2,2);
scatter(X(:),Y(:),50,tc(:),'filled');
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(ax2,flipud(cmap))
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([0.15,4.15]);
yticks(0.15:2:4.15)
colorbar;
clim([0,800])


xedges = -14:0.4:-10;
yedges = -4.15:0.4:-0.15;
[X,Y] = meshgrid((xedges(1:end-1)+xedges(2:end))/2,(yedges(1:end-1)+yedges(2:end))/2);
[hc_start,xedges,yedges,xbin,ybin] = histcounts2(trajX(1,:),trajY(1,:),xedges,yedges);
hc_start_success = histcounts2(trajX(1,success),trajY(1,success),xedges,yedges);
tc = zeros(size(X));
for i = 1:length(totTime)
    if success(i)
        tc(xbin(i),ybin(i)) = tc(xbin(i),ybin(i)) + totTime(i);
    end
end
tc = tc'./hc_start_success';
sr = hc_start_success'./hc_start';
subplot(2,2,3),scatter(X(:),Y(:),50,sr(:),'filled');
% hc_start = [hc_start zeros(size(hc_start,1),1)];hc_start = [hc_start;zeros(1,size(hc_start,2))];
% hc_start_success = [hc_start_success zeros(size(hc_start_success,1),1)];
% hc_start_success = [hc_start_success;zeros(1,size(hc_start_success,2))];
% subplot(2,2,3),h_start = pcolor(X',Y',hc_start_success./hc_start);
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([-4.15,-0.15]);
yticks(-4.15:2:-0.15)
colorbar;
clim([0,1])

ax4=subplot(2,2,4);
scatter(X(:),Y(:),50,tc(:),'filled');
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(ax4,flipud(cmap))
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([-4.15,-0.15]);
yticks(-4.15:2:-0.15)
colorbar;
clim([0,800])
sgtitle(policy_name);
exportgraphics(gcf,['./savedFigs/grading_analysis_' policy_name env '_ratetime.eps'],'ContentType','vector')
%%
figure('Position',[705 13 886 1309]);
obs_x = squeeze(observation(1,:,:)); obs_y = squeeze(observation(2,:,:));
obs_x = obs_x(:); obs_y = obs_y(:);
obs_x = obs_x(abs(obs_x)>0);obs_y = obs_y(abs(obs_y)>0);
xbin = round((max(obs_x)-min(obs_x))/0.13);
ybin = round((max(obs_y)-min(obs_y))/0.13);
[hc_obs,xedges,yedges] = histcounts2(obs_x,obs_y,[xbin,ybin],'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges); X = X'; Y= Y';
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax = subplot(4,1,1); h_obs = pcolor(X,Y,hc_obs);
hold on, plot(X(hc_obs==0),Y(hc_obs==0),'k.','MarkerSize',2)
axis equal;
xlim([min(xedges),max(xedges)]);
ylim([min(yedges),max(yedges)]);
xlabel('$\Delta x$');ylabel('$\Delta y$');
% map = flip(gray);
map = ones(400,3);
map(:,1) = linspace(1,0.84,400);
map(:,2) = linspace(1,0.16,400);
map(:,3) = linspace(1,0.16,400);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

obs_u = squeeze(observation(4,:,:)); obs_v = squeeze(observation(5,:,:));
obs_u = obs_u(:); obs_v = obs_v(:);
obs_u = obs_u(abs(obs_u)>0);obs_v = obs_v(abs(obs_v)>0);
ubin = round((max(obs_u)-min(obs_u))/0.018);
vbin = round((max(obs_v)-min(obs_v))/0.018);
[hc_obs,xedges,yedges] = histcounts2(obs_u,obs_v,[ubin,vbin],'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges); X = X'; Y= Y';
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax=subplot(4,1,2); h_obs = pcolor(X,Y,hc_obs);
hold on, plot(X(hc_obs==0),Y(hc_obs==0),'k.','MarkerSize',2)
axis equal;
xlim([min(xedges),max(xedges)]);
ylim([min(yedges),max(yedges)]);
xlabel('$u$');ylabel('$v$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
% map = flip(gray);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

obs_theta = squeeze(observation(3,:,:)); obs_theta = obs_theta(:); 
obs_theta = obs_theta(abs(obs_theta)>0);
ax=subplot(4,1,3); h_obs = histogram(obs_theta,'Normalization','pdf');
xlabel('$\theta$');ylabel('PDF');
xlim([-pi,pi])
xticks(-pi:pi/2:pi);
h_obs.EdgeAlpha = 0;

actions = 4*action(:);
% subplot(2,2,4),h_act = histfit(action(abs(action)>0));
subplot(4,1,4),h_act = histogram(actions(abs(actions)>0),'Normalization','probability');
xlabel('$\dot\theta$');ylabel('PDF');
xlim([-4,4]);
ylim([0, 1.5]);
sgtitle(policy_name);
h_act(1).EdgeAlpha = 0;
exportgraphics(gcf,['./savedFigs/grading_analysis_' policy_name env '_obsact.eps'],'ContentType','vector')
%% grade egoLRGradreduced1 in egoLRGradCFD
policy_name = 'egoLRGradreduced1';
load([root_dir policy_name '/grading_results_ego2sensorLRGradCFD.mat']);
% load([root_dir 'egoLRGrad1/grading_results_egoLRGradreduced.mat']);
success = reward>50;
figure('position',[297 850 630 472]);
sgtitle('egoLRGrad1')
cmap = cbrewer('seq','YlGn',400,'linear');
colormap(cmap)
xedges = -14:0.4:-10;
yedges = 0.15:0.4:4.15;
[X,Y] = meshgrid((xedges(1:end-1)+xedges(2:end))/2,(yedges(1:end-1)+yedges(2:end))/2);
[hc_end,xedges,yedges,xbin,ybin] = histcounts2(target(:,1),target(:,2),xedges,yedges);
hc_end_success = histcounts2(target(success,1),target(success,2),xedges,yedges);
tc = zeros(size(X));
for i = 1:length(totTime)
    if success(i)
        tc(xbin(i),ybin(i)) = tc(xbin(i),ybin(i)) + totTime(i);
    end
end
tc = tc'./hc_end_success';
sr = hc_end_success'./hc_end';
subplot(2,2,1),scatter(X(:),Y(:),50,sr(:),'filled');
% hc_end = [hc_end zeros(size(hc_end,1),1)];hc_end = [hc_end;zeros(1,size(hc_end,2))];
% hc_end_success = [hc_end_success zeros(size(hc_end_success,1),1)];
% hc_end_success = [hc_end_success;zeros(1,size(hc_end_success,2))];
% subplot(2,2,1),h_end = pcolor(X',Y',hc_end_success./hc_end);
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([0.15,4.15]);
yticks(0.15:2:4.15)
colorbar;
caxis([0,1])


ax2=subplot(2,2,2);
scatter(X(:),Y(:),50,tc(:),'filled');
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(ax2,flipud(cmap))
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([0.15,4.15]);
yticks(0.15:2:4.15)
colorbar;
caxis([0,800])


xedges = -14:0.4:-10;
yedges = -4.15:0.4:-0.15;
[X,Y] = meshgrid((xedges(1:end-1)+xedges(2:end))/2,(yedges(1:end-1)+yedges(2:end))/2);
[hc_start,xedges,yedges,xbin,ybin] = histcounts2(trajX(1,:),trajY(1,:),xedges,yedges);
hc_start_success = histcounts2(trajX(1,success),trajY(1,success),xedges,yedges);
tc = zeros(size(X));
for i = 1:length(totTime)
    if success(i)
        tc(xbin(i),ybin(i)) = tc(xbin(i),ybin(i)) + totTime(i);
    end
end
tc = tc'./hc_start_success';
sr = hc_start_success'./hc_start';
subplot(2,2,3),scatter(X(:),Y(:),50,sr(:),'filled');
% hc_start = [hc_start zeros(size(hc_start,1),1)];hc_start = [hc_start;zeros(1,size(hc_start,2))];
% hc_start_success = [hc_start_success zeros(size(hc_start_success,1),1)];
% hc_start_success = [hc_start_success;zeros(1,size(hc_start_success,2))];
% subplot(2,2,3),h_start = pcolor(X',Y',hc_start_success./hc_start);
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([-4.15,-0.15]);
yticks(-4.15:2:-0.15)
colorbar;
caxis([0,1])


ax4=subplot(2,2,4);
scatter(X(:),Y(:),50,tc(:),'filled');
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(ax4,flipud(cmap))
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([-4.15,-0.15]);
yticks(-4.15:2:-0.15)
colorbar;
caxis([0,800])
sgtitle(policy_name);
exportgraphics(gcf,['./savedFigs/grading_analysis_' policy_name 'InCFD_ratetime.eps'],'ContentType','vector')

figure('Position',[960 624 1029 689]);
obs_x = squeeze(observation(1,:,:)); obs_y = squeeze(observation(2,:,:));
obs_x = obs_x(:); obs_y = obs_y(:);
obs_x = obs_x(abs(obs_x)>0);obs_y = obs_y(abs(obs_y)>0);
[hc_obs,xedges,yedges] = histcounts2(obs_x,obs_y,'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges);
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax = subplot(2,2,1); h_obs = pcolor(X',Y',hc_obs);
axis equal;
xlabel('$\Delta x$');ylabel('$\Delta y$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
map = flip(gray);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

obs_u = squeeze(observation(3,:,:)); obs_v = squeeze(observation(4,:,:));
obs_u = obs_u(:); obs_v = obs_v(:);
obs_u = obs_u(abs(obs_u)>0);obs_v = obs_v(abs(obs_v)>0);
[hc_obs,xedges,yedges] = histcounts2(obs_u,obs_v,'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges);
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax=subplot(2,2,3); h_obs = pcolor(X',Y',hc_obs);
axis equal;
xlabel('$u$');ylabel('$v$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
map = flip(gray);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

obs_dudy = squeeze(observation(5,:,:)); obs_dvdy = squeeze(observation(6,:,:));
obs_dudy = obs_dudy(:); obs_dvdy = obs_dvdy(:);
obs_dudy = obs_dudy(abs(obs_dudy)>0);obs_dvdy = obs_dvdy(abs(obs_dvdy)>0);
[hc_obs,xedges,yedges] = histcounts2(obs_dudy,obs_dvdy,'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges);
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax=subplot(2,2,2);h_obs = pcolor(X',Y',hc_obs);
axis equal;
xlabel('$\partial u/\partial y$');ylabel('$\partial v/\partial y$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
map = flip(gray);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

action = 4*action(:);
subplot(2,2,4),h_act = histogram(action(abs(action)>0),'Normalization','pdf');
xlabel('$\dot\theta$');ylabel('PDF');
xlim([-4,4]);
h_act.EdgeAlpha = 0;
exportgraphics(gcf,['./savedFigs/grading_analysis_' policy_name 'InCFD_obsact.eps'],'ContentType','vector')
%% grading results of egoLRGradreduced1
policy_name = 'egoLRGradreduced1';
load([root_dir policy_name '/grading_results_egoLRGradreduced.mat']);

success = reward>50;
figure('position',[297 850 630 472]);
sgtitle('egoLRGradreduced1')
cmap = cbrewer('seq','YlGn',400,'linear');
colormap(cmap)
xedges = -14:0.4:-10;
yedges = 0.15:0.4:4.15;
[X,Y] = meshgrid((xedges(1:end-1)+xedges(2:end))/2,(yedges(1:end-1)+yedges(2:end))/2);
[hc_end,xedges,yedges,xbin,ybin] = histcounts2(target(:,1),target(:,2),xedges,yedges);
hc_end_success = histcounts2(target(success,1),target(success,2),xedges,yedges);
tc = zeros(size(X));
for i = 1:length(totTime)
    if success(i)
        tc(xbin(i),ybin(i)) = tc(xbin(i),ybin(i)) + totTime(i);
    end
end
tc = tc'./hc_end_success';
sr = hc_end_success'./hc_end';
subplot(2,2,1),scatter(X(:),Y(:),50,sr(:),'filled');
% hc_end = [hc_end zeros(size(hc_end,1),1)];hc_end = [hc_end;zeros(1,size(hc_end,2))];
% hc_end_success = [hc_end_success zeros(size(hc_end_success,1),1)];
% hc_end_success = [hc_end_success;zeros(1,size(hc_end_success,2))];
% subplot(2,2,1),h_end = pcolor(X',Y',hc_end_success./hc_end);
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([0.15,4.15]);
yticks(0.15:2:4.15)
colorbar;
clim([0,1])

ax2=subplot(2,2,2);
scatter(X(:),Y(:),50,tc(:),'filled');
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(ax2,flipud(cmap))
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([0.15,4.15]);
yticks(0.15:2:4.15)
colorbar;
clim([0,800])


xedges = -14:0.4:-10;
yedges = -4.15:0.4:-0.15;
[X,Y] = meshgrid((xedges(1:end-1)+xedges(2:end))/2,(yedges(1:end-1)+yedges(2:end))/2);
[hc_start,xedges,yedges,xbin,ybin] = histcounts2(trajX(1,:),trajY(1,:),xedges,yedges);
hc_start_success = histcounts2(trajX(1,success),trajY(1,success),xedges,yedges);
tc = zeros(size(X));
for i = 1:length(totTime)
    if success(i)
        tc(xbin(i),ybin(i)) = tc(xbin(i),ybin(i)) + totTime(i);
    end
end
tc = tc'./hc_start_success';
sr = hc_start_success'./hc_start';
subplot(2,2,3),scatter(X(:),Y(:),50,sr(:),'filled');
% hc_start = [hc_start zeros(size(hc_start,1),1)];hc_start = [hc_start;zeros(1,size(hc_start,2))];
% hc_start_success = [hc_start_success zeros(size(hc_start_success,1),1)];
% hc_start_success = [hc_start_success;zeros(1,size(hc_start_success,2))];
% subplot(2,2,3),h_start = pcolor(X',Y',hc_start_success./hc_start);
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([-4.15,-0.15]);
yticks(-4.15:2:-0.15)
colorbar;
clim([0,1])

ax4=subplot(2,2,4);
scatter(X(:),Y(:),50,tc(:),'filled');
cmap = cbrewer('seq','BuPu',400,'linear');
colormap(ax4,flipud(cmap))
axis equal;
xlabel('x');ylabel('y');
xlim([-14,-10]);ylim([-4.15,-0.15]);
yticks(-4.15:2:-0.15)
colorbar;
clim([0,800])
sgtitle(policy_name);
exportgraphics(gcf,['./savedFigs/grading_analysis_' policy_name '_ratetime.eps'],'ContentType','vector')
%%
figure('Position',[705 13 886 1309]);
obs_x = squeeze(observation(1,:,:)); obs_y = squeeze(observation(2,:,:));
obs_x = obs_x(:); obs_y = obs_y(:);
obs_x = obs_x(abs(obs_x)>0);obs_y = obs_y(abs(obs_y)>0);
xbin = round((max(obs_x)-min(obs_x))/0.24);
ybin = round((max(obs_y)-min(obs_y))/0.24);
[hc_obs,xedges,yedges] = histcounts2(obs_x,obs_y,[xbin, ybin],'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges);
X = X'; Y= Y';
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax = subplot(4,1,1); h_obs = pcolor(X,Y,hc_obs);
hold on, plot(X(hc_obs==0),Y(hc_obs==0),'k.','MarkerSize',2)
axis equal;
xlim([min(xedges),max(xedges)]);
ylim([min(yedges),max(yedges)]);
xlabel('$\Delta x$');ylabel('$\Delta y$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
% map = flip(hot);
map = ones(400,3);
map(:,1) = linspace(1,0.84,400);
map(:,2) = linspace(1,0.16,400);
map(:,3) = linspace(1,0.16,400);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

obs_u = squeeze(observation(3,:,:)); obs_v = squeeze(observation(4,:,:));
obs_u = obs_u(:); obs_v = obs_v(:);
obs_u = obs_u(abs(obs_u)>0);obs_v = obs_v(abs(obs_v)>0);
ubin = round((max(obs_u)-min(obs_u))/0.035);
vbin = round((max(obs_v)-min(obs_v))/0.035);
[hc_obs,xedges,yedges] = histcounts2(obs_u,obs_v,[ubin,vbin],'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges); X = X'; Y= Y';
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax=subplot(4,1,2); h_obs = pcolor(X,Y,hc_obs);
hold on, plot(X(hc_obs==0),Y(hc_obs==0),'k.','MarkerSize',2)
axis equal;
xlim([min(xedges),max(xedges)]);
ylim([min(yedges),max(yedges)]);
xlabel('$u$');ylabel('$v$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
% map = flip(gray);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

obs_dudy = squeeze(observation(5,:,:)); obs_dvdy = squeeze(observation(6,:,:));
obs_dudy = obs_dudy(:); obs_dvdy = obs_dvdy(:);
flag = (abs(obs_dudy)+abs(obs_dvdy))>0;
obs_dudy = obs_dudy(flag);obs_dvdy = obs_dvdy(flag);
limit_dudy = 2;
limit_dvdy = 0.9;
flag = abs(obs_dudy)<limit_dudy & abs(obs_dvdy)<limit_dvdy;
obs_dudy = obs_dudy(flag);obs_dvdy = obs_dvdy(flag);
% dudybin = round((max(obs_dudy)-min(obs_dudy))/0.02);
% dvdybin = round((max(obs_dvdy)-min(obs_dvdy))/0.02);
% dudybin = round(2*limit_dudy/0.02);
% dvdybin = round(2*limit_dvdy/0.02);
xedges = -limit_dudy:0.02:limit_dudy;
yedges = -limit_dvdy:0.02:limit_dvdy;
[hc_obs,xedges,yedges] = histcounts2(obs_dudy,obs_dvdy,xedges,yedges,'Normalization','pdf');
[X,Y] = meshgrid(xedges,yedges); X = X'; Y= Y';
hc_obs = [hc_obs zeros(size(hc_obs,1),1)];hc_obs = [hc_obs;zeros(1,size(hc_obs,2))];

ax=subplot(4,1,3);h_obs = pcolor(X,Y,hc_obs);
% Z = ones(size(X));
hold on, plot(X(hc_obs==0),Y(hc_obs==0),'k.','MarkerSize',2)
axis equal;
% xlim([min(xedges),max(xedges)]);
% ylim([min(yedges),max(yedges)]);
xlim([-limit_dudy,limit_dudy]);
ylim([-limit_dvdy,limit_dvdy]);
xlabel('$\partial u/\partial y$');ylabel('$\partial v/\partial y$');
% xlim([-14,-10]);ylim([0.15,4.15]);
% yticks(0.15:2:4.15)
% map = flip(gray);
colormap(ax,map)
colorbar;
h_obs.EdgeAlpha = 0;

actions = 4*action(:);
% subplot(2,2,4),h_act = histfit(action(abs(action)>0));
subplot(4,1,4),h_act = histogram(actions(abs(actions)>0),'Normalization','pdf');
xlabel('$\dot\theta$');ylabel('PDF');
xlim([-4,4]);
ylim([0,1.5]);
h_act(1).EdgeAlpha = 0;
exportgraphics(gcf,['./savedFigs/grading_analysis_' policy_name '_obsact.eps'],'ContentType','vector')