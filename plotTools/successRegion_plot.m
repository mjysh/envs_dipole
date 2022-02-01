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
T = readlines([dir 'success_region.txt']);
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

% reward = normalize(reward,'range');

for i = 1:8:length(initY)
    sr(i) = sum(reward(i:i+7)>50)/8;
end
% reward = (reward+10)/220;
%%
figure('Position',[960 848 640 284]);
[bg,map] = imread("movie2000.png","png");hold on;
image([-24,8],[-8,8],ind2rgb(bg,map));
plot(targetX-8,targetY,'*',Color=[50/255,100/255,50/255]);

% for i = 1:length(T)-2
% p = plot([initX(i), initX(i)+0.3*cos(initTheta(i))]-8, [initY(i), initY(i) + 0.3*sin(initTheta(i))],'Color',[reward(i),0,0],'LineWidth',1);
% end
colormap parula
imagesc([min(initX),max(initX)]-8,[min(initY),max(initY)],reshape(sr(1:8:end),10,[]),'AlphaData',0.8);
axis equal;
colorbar("Ticks",[],'Location','westoutside')
xlim([-23.5,0]);
ylim([-6,6]);
axis off
the = 0:pi/200:pi*2;
plot(-12+cos(the),-2.15+sin(the),'k');
