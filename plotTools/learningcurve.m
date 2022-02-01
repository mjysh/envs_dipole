%%
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
T = readlines([dir 'agent_00_rank_000_cumulative_rewards.dat']);

reward = zeros(length(T)-1,1);

for i = 1:length(T)-1
    resultStrings = split(T(i));
    reward(i) = str2double(resultStrings(end));
end

%%
figure("Position", [960 1061 363 252]);
% plot(targetX,targetY,'b*');
p(1) = plot(reward,'.','MarkerSize',1);hold on
p(2) = plot(movmean(reward,200),'r','LineWidth',1);
% axis equal;
% xlim([-24,0]);
ylim([-10,210]);
ylabel('reward');
xlabel('episodes');