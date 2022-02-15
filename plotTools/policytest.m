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
T = readlines([dir 'policytest.']);

obs = zeros(length(T)-1,1);
action = zeros(length(T)-1,1);

for i = 1:length(T)-1
    resultStrings = split(T(i));
    obs(i) = str2double(resultStrings(1));
    a = char(resultStrings(end));
    action(i) = str2double(a(1:end));
end

%%
figure("Position", [960 1061 363 252]);
% plot(targetX,targetY,'b*');
p(1) = plot(obs,action,'.','MarkerSize',5);hold on
% axis equal;
% xlim([-24,0]);
% ylim([-10,210]);
ylabel('action');
xlabel('observation');