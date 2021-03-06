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
dir = './';
load([dir 'policyTest.mat']);
%% labframe
deltaX = observations(:,1);
deltaY = observations(:,2);
orientation = observations(:,3);
flowU = observations(:,4);
flowV = observations(:,5);

% theta == 0

%% flow field
v = VideoWriter('lab.avi');
open(v);
figure('Position',[519 880 901 402]);
% subplot(2,1,1)
title('lab frame observation, $\theta=-180^\circ$')
[bg,map] = imread("movie0000.png","png");hold on;
image([-24,8],[-8,8],ind2rgb(bg,map));
targetX = -12;
targetY = 2;
plot(-12,2,'*',Color=[50/255,100/255,50/255]);
%
% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
colormap parula
im=imagesc([-23,-1],[-5,5.25],reshape(actions(1:36:end),42,[]),'AlphaData',0.8,[-1,1]);hold on;
axis equal;
% colorbar("Ticks",[],'Location','westoutside')
cb = colorbar('Location','westoutside');
cb.Label.Interpreter = 'latex';
cb.Label.String = '$\dot\theta$';
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
% plot training region
plot(-12+2*cos(the),-2.15+2*sin(the),'k');
frame = getframe(gcf);
writeVideo(v,frame);
% plot(-observations(end/4*3+1,1)-12,-observations(end/4*3+1,2)+2,'ko')
for i = 2:36
    pause(0.5)
    im.CData = reshape(actions(i:36:end),42,[]);
    title([num2str() 'observation, ','$\theta =', num2str(i/36*360-190),'^\circ$'])
    xlim([-23.5,0]);
    ylim([-6,6]);
    drawnow;
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v);

% subplot(2,1,2);
figure();
% test point
testX = -18;
testY = -1;
plot(-18,-1,'bo');
plot([0:pi/18:pi-pi/18 -pi:pi/18:-pi/18],actions(logical((abs(deltaX-targetX+testX)<1e-3).*(abs(deltaY-targetY+testY)<1e-3))),'.');
xlim([-pi,pi])
hold on, plot([-pi,pi],[0,0],'--','Color',[0.3,0.3,0.3])
%% 
targetAngle = atan2(deltaY, deltaX);
flowAngle = atan2(flowV, flowU);
figure;
action_normal = (actions - min(actions));
action_normal = action_normal/max(action_normal);
map = jet(512);
colors = round(action_normal*511)+1;
scatter(wrapToPi(targetAngle),orientation,15, map(colors,:),AlphaData=0.8);
xlim([-pi,pi]);
xticks(-pi:pi/2:pi);
xlabel("atan2($\Delta x/\Delta y$)");
xticklabels(["$-\pi$","$-\pi/2$","$0$","$\pi/2$","$\pi$"]);
ylim([-pi,pi]);
yticks(-pi:pi/2:pi);
yticklabels(["$-\pi$","$-\pi/2$","$0$","$\pi/2$","$\pi$"]);
ylabel("orientation");
colormap jet
colorbar;
%% egocentric frame
deltaX = observations(:,1);
deltaY = observations(:,2);
flowU = observations(:,3);
flowV = observations(:,4);
Ugrad2 = observations(:,5);
Vgrad2 = observations(:,6);
% theta == 0

%% flow field
figure('Position',[955 263 900 857]);
subplot(2,1,1)
title('lab frame observation, $\theta=-180^\circ$')
[bg,map] = imread("movie0000.png","png");hold on;
image([-24,8],[-8,8],ind2rgb(bg,map));
targetX = -12;
targetY = 2;
plot(-12,2.25,'*',Color=[50/255,100/255,50/255]);
%
% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
colormap parula
im=imagesc([-23,-1],[-5.5,5.75],reshape(actions(1:36:end),46,[]),'AlphaData',0.8,[-1,1]);hold on;
axis equal;
% colorbar("Ticks",[],'Location','westoutside')
cb = colorbar('Location','westoutside');
cb.Label.Interpreter = 'latex';
cb.Label.String = '$\dot\theta$';
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
% plot training region
plot(-12+2*cos(the),-2.15+2*sin(the),'k');

% plot(-observations(end/4*3+1,1)-12,-observations(end/4*3+1,2)+2,'ko')
for i = 2:36
    pause(0.5)
    im.CData = reshape(actions(i:36:end),46,[]);
    title(['lab frame observation, ','$\theta =', num2str(i/36*360-190),'^\circ$'])
    drawnow;
end


subplot(2,1,2);
% test point
testX = -18;
testY = -1;
plot(-18,-1,'bo')
plot([0:pi/18:pi-pi/18 -pi:pi/18:-pi/18],actions(logical((abs(deltaX-targetX+testX)<1e-3).*(abs(deltaY-targetY+testY)<1e-3))),'.');
xlim([-pi,pi])
hold on, plot([-pi,pi],[0,0],'--','Color',[0.3,0.3,0.3])
