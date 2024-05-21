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
root_dir = '/home/yusheng/smarties/apps/dipole_adapt/paper_new/';
RLsetting_list = {'geo';
    'egoLRGrad';
    'egolimited';
    'geoflowblind';
    'georeduced';
    'egoLRGradreduced';
    'egoDirLRGradreduced';
    'egolimitedreduced';
    'transfer_exp';
    'egoDirLRGradCFD'};
policies = struct('name',RLsetting_list,'count',0,'folder_name',[],'training_length',[],'reward_final',[],'rewards',[],'converge_time',[],'timesteps',[],'grading',[]);
all_content = dir(root_dir);
for i = 1:length(all_content)
    for l = 1:length(RLsetting_list)
        if isfile([root_dir all_content(i).name]) && endsWith(all_content(i).name, [RLsetting_list{l} '_grade.txt'])
            t = readtable([root_dir all_content(i).name]);
            policies(l).grading = t.successRate;
            break;
        elseif startsWith(all_content(i).name, RLsetting_list{l}) && ~isnan(str2double(all_content(i).name(length(RLsetting_list{l})+1:end)))
            policies(l).count = policies(l).count + 1;
            policies(l).folder_name{end+1} = all_content(i).name;
            break;
        end
    end
end
%%
average_window = 500;
for l = 1:length(RLsetting_list)

    %     reward = zeros(length(t_q),n_end-n_start+1);
    %         convergence_time = zeros(n_end-n_start+1,1);
    for i = 1:policies(l).count
        [reward_raw,time_step] = read_one_instance([root_dir policies(l).folder_name{i}]);
        firstnonzero = find(time_step,1);
        lastnonzero = find(time_step,1,'last');
        policies(l).training_length(end+1) = time_step(lastnonzero);
        policies(l).reward_final(end+1) = mean(reward_raw(lastnonzero-average_window+1:lastnonzero));
        t_q = linspace(0,time_step(lastnonzero),101);
        policies(l).timesteps(i) = time_step(lastnonzero);
        policies(l).rewards{end+1} = zeros(length(t_q),1);
        policies(l).rewards{end}(1) = mean(reward_raw(1:firstnonzero-1));
        policies(l).rewards{end}(2:end) = interp1(time_step(firstnonzero:lastnonzero), ...
            movmean(reward_raw(firstnonzero:lastnonzero),average_window),t_q(2:end),'linear','extrap');
        convergence_index = find(abs(policies(l).rewards{end} - policies(l).reward_final(end))<10,1);
        if (convergence_index == 101) || isempty(convergence_index)
            policies(l).converge_time(end+1) = t_q(end);
        else
            policies(l).converge_time(end+1) = t_q(convergence_index+1);
        end
    end
end
save("learning_data.mat","policies")
%%
[reward_raw,time_step] = read_one_instance([root_dir 'geo1']);
figure,plot(time_step,reward_raw,'.','MarkerSize',1);
xlabel('time steps');
ylabel('cumulative reward');
figure,plot(reward_raw,'.','MarkerSize',1);
xlabel('episodes');
ylabel('cumulative reward');
%%
average_window = 500;
firstnonzero = find(time_step,1);
lastnonzero = find(time_step,1,'last');
t_q = linspace(0,time_step(lastnonzero),101);
rewards = zeros(length(t_q),1);
reward_final = mean(reward_raw(lastnonzero-average_window+1:lastnonzero));
rewards(1) = mean(reward_raw(1:firstnonzero-1));
rewards(2:end) = interp1(time_step(firstnonzero:lastnonzero), ...
    movmean(reward_raw(firstnonzero:lastnonzero),average_window),t_q(2:end),'linear','extrap');
figure,plot(t_q,rewards);
xlabel('time steps');
ylabel('cumulative reward (moving average interp)');
convergence_index = find(abs(rewards - reward_final)<10,1);
hold on;
plot([t_q(convergence_index) t_q(convergence_index)],[0,rewards(convergence_index)],'r--')

%%
load('learning_data.mat')
for l = 6:length(policies)
    figure;
    l
    ax1 = subplot('Position',[0.12,0.46,0.48,0.48]);hold on;
    ax2 = subplot('Position',[0.12,0.1,0.48,0.23]);
    ax3 = subplot('Position',[0.65,0.46,0.25,0.48]);
    ax4 = subplot('Position',[0.65,0.1,0.25,0.23]);
    
    ylabel(ax1,'rewards');
    for instance = 1:policies(l).count
        t_q = linspace(0,policies(l).timesteps(instance),101);
        plot(ax1,t_q,policies(l).rewards{instance},'k','LineWidth',1);
        plot(ax1,[policies(l).converge_time(instance) policies(l).converge_time(instance)],[0,200],'r--','LineWidth',0.5)
    end
%     max_timesteps = max(policies(l).timesteps);
    xlim(ax1,[0,2e7]);
    ylim(ax1,[-10,210]);
    sgtitle(policies(l).name)
    
    histogram(ax2,policies(l).converge_time,linspace(0,2e7,21));
    xlim(ax2,[0,2e7]);
    xlabel(ax2,'time steps');
    ylabel(ax2,'count of convergence')
    
    histogram(ax3,policies(l).reward_final,linspace(-10,210,23),'Orientation','horizontal');
    ylim(ax3,[-10,210]);
    xlabel(ax3,'count');
    ylabel(ax3,'final mean reward');
    
    
    ax2.FontSize = 12;
    ax3.FontSize = 12;

    plot(ax4,policies(l).grading*100,'b^','MarkerSize',4);
    ylabel(ax4,'grade');
    ylim([0,100]);
    set(ax3,'YAxisLocation','right')
    set(ax4,'YAxisLocation','right')
    exportgraphics(gcf,['./savedFigs/learning' policies(l).name '.eps'],'ContentType','vector')
end
%%
grade_mean = zeros(length(policies),1);
grade_std = zeros(length(policies),1);
grade_best = zeros(length(policies),1);
grade_worst = zeros(length(policies),1);
policy_count = zeros(length(policies),1);
for l = 1:length(policies)-2
    grade_mean(l) = 100*mean(policies(l).grading);
    grade_std(l) = 100*std(policies(l).grading);
    policy_count(l) = policies(l).count;
    grade_best(l) = 100*max(policies(l).grading);
    grade_worst(l) = 100*min(policies(l).grading);
end

trainingStatistics = table(RLsetting_list,policy_count, grade_best,grade_worst,grade_mean,grade_std);
writetable(trainingStatistics)
%%
train_dir = '/home/yusheng/smarties/apps/dipole_adapt/paper_new/egoLRGrad';
train_dir = '/home/yusheng/smarties/apps/dipole_adapt/paper_new/georeduced';
n_best = 1;
max_timestep = 2e7;
t_q = linspace(0,max_timestep,101);
%%
start_index = 4;
end_index = 4;
average_window = 500;
reward = read_learning_log(train_dir, start_index, end_index,t_q,average_window);
%% all curves
% figure("Position", [960 1061 363 252]);
plot(t_q,reward,'-','color',[0.2,0.3,0.7],'LineWidth',0.25); hold on;
plot(t_q,reward(:,n_best)','-','color',[0.7,0.3,0.2],'LineWidth',1);

ylim([-10,210]);
ylabel('reward');
xlabel('time steps');
%%
reward_std = zeros(1,100);
reward_min = zeros(1,100);
reward_max = zeros(1,100);
for i = 1:100
    reward_min(i) = min(reward(i:end));
    reward_max(i) = max(reward(i:end));
    reward_std(i) = std(reward(i:end));
end
figure,plot(reward_std/200);
hold on, plot((max(reward)-reward_min)/200);
legend
%% range
figure("Position", [960 1061 363 252]);
x_axis = t_q;lColor = [0.2 0.6 0.7];aColor = [0.5 0.75 0.85];lWidth = 1.5;
[patch_s1,l_s1] = plot_shadederrorbar(reward,x_axis,lColor,aColor,lWidth);

ylim([-10,210]);
ylabel('reward');
xlabel('time steps');
%% single
figure("Position", [960 1061 363 252]);
[reward_raw,time_step] = read_one_instance([train_dir num2str(n_best)]);
plot(time_step,reward_raw,'.','MarkerSize',0.1,'color',[0.5 0.75 0.85],'LineWidth',0.5);hold on
plot(time_step,movmean(reward_raw,100),'-','color',[0.2,0.3,0.7],'LineWidth',0.5);
ylim([-10,210]);
ylabel('reward');
xlabel('time steps');
%%
function [reward,reward_raw,time_step] = read_learning_log(train_dir, n_start, n_end,t_q,average_window)
reward = zeros(length(t_q),n_end-n_start+1);
for k = 1:n_end-n_start+1
    [reward_raw,time_step] = read_one_instance([train_dir num2str(k)]);
    firstnonzero = find(time_step,1);
    lastnonzero = find(time_step,1,'last');
    reward(1,k) = mean(reward_raw(1:firstnonzero-1));
    reward(2:end,k) = interp1(time_step(firstnonzero:lastnonzero),movmean(reward_raw(firstnonzero:lastnonzero),average_window),t_q(2:end),'linear','extrap');
end

end
function [reward_raw,time_step] = read_one_instance(train_path)
T = readlines([train_path '/agent_00_rank_000_cumulative_rewards.dat']);
reward_raw = zeros(1,length(T)-1);
time_step = zeros(1,length(T)-1);
valid = true(1,length(T)-1);
for i = 1:length(T)-1
    resultStrings = split(T(i));

    reward_raw(i) = str2double(resultStrings(end));
    time_step(i) = str2double(resultStrings(2));
    if (time_step(i)>0) && (time_step(i) <= time_step(i-1))
        last_index = find(time_step(1:i)<time_step(i),1,'last');
        valid(last_index+1:i-1) = false;
    end
end
reward_raw = reward_raw(valid);
time_step = time_step(valid);
end