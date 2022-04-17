%%
close all;
% clear;
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
CFD = load('CFDData.mat');
%%
path = 'egoGradLR/trajectory12-12.mat';
go(path,'egograd12-12randtest2', CFD);
% path = 'lab/trajectory12-12.mat';
% go(path,'lab12-12', CFD);
% path = 'egoGradLR/trajectory.mat';
% go(path,'ego1', CFD);
% path = 'egoGradLR/trajectory2.mat';
% go(path,'ego2', CFD);
%%
function go(path,savename, CFD)
load(['/home/yusheng/navigation_envs/dipole_new/plotTools/' path], 'states','time','target')
options = optimoptions("fmincon",'Display','iter-detailed','MaxFunctionEvaluations',1e8, ...
    'MaxIterations',1e4,'Algorithm','interior-point', ...
    'PlotFcn',@(x,optimValues,state,varargin) optimplottraj(savename,x,optimValues,state,varargin));
% N = height(states);
N = 350;
% target = [-12, 2.25];
% initPos = [-15, -2.25, -pi];
initPos = states(1,:);
xmin = -24; xmax = 0;
ymax = 5; ymin = -5;
dtmax = 0.2;
dtmin = 0.005;
thetamax = pi; thetamin = -pi;
thetadotmax = 4;
lb = [ones(N,1)*xmin;
    ones(N,1)*ymin;
    thetamin;
    ones(N-1,1)*(-thetadotmax);
    dtmin];
ub = [ones(N,1)*xmax;
    ones(N,1)*ymax;
    thetamax;
    ones(N-1,1)*thetadotmax;
    dtmax];

A = [];
b = [];
Aeq = zeros(3,3*N+1);
Aeq(1,1) = 1;
Aeq(2,1+N) = 1;
Aeq(3,1+2*N) = 1;
% Aeq(3,end) = 1;
% Aeq(4,2*N) = 1;
beq = [initPos(1); initPos(2); wrapToPi(initPos(3))];

% x0 = [states(:,1); 
%     states(:,2); 
%     wrapToPi(initPos(3));
%     diff(states(:,3))/0.1;
%     0.1];
x0 = [linspace(initPos(1),target(1),N)'; 
    linspace(initPos(2), target(2),N)'; 
    initPos(3);
    ones(N-1,1)*(3*pi/2/(N-1));
    0.1];
x0 = min(ub - 0.00001,x0);
x0 = max(lb + 0.00001,x0);

%%

% [UUU, VVV, OOO, XMIN, XMAX, YMIN, YMAX, cfd_framerate, time_span, cf] = load_data(4.5);
% mdic = py.dict(pyargs('UUU', UUU, 'VVV', VVV, 'OOO', OOO, ...
%     'XMIN', XMIN, 'XMAX', XMAX, 'YMIN', YMIN, 'YMAX',YMAX, ...
%     'cfd_framerate', cfd_framerate, 'time_span', time_span));
% smat = py.importlib.import_module('scipy.io');
% smat.savemat('CFDData.mat', mdic);
%%
% UUU = AllCellsToDouble(UUU);
% VVV = AllCellsToDouble(VVV);
% OOO = AllCellsToDouble(OOO);
% XMIN = ListsToCells(XMIN);
% XMAX = ListsToCells(XMAX);
% YMIN = ListsToCells(YMIN);
% YMAX = ListsToCells(YMAX);
% cfd_framerate = double(cfd_framerate);
% terminate(pyenv)
%%
% time = 0.116;
% posX = -13;
% posY = 1.24;
% [U_interp,V_interp,O_interp] = adapt_time_interp(UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX,cfd_framerate,time,posX, posY)
%%
fun = @objective;
nonlcon = @(x) dynamiccon(x, target, N,CFD, time(1));
[x,fval,exitflag,output] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% save([savename 'optimtraj.mat'],"fval","output");
plot(x(1:N),x(N+1:2*N),'.--'); hold on;
plot(x(1),x(N+1),'r*');
plot(x(N),x(2*N),'b*');
plot(x0(1:N),x0(N+1:2*N),'k--');
title(['timestep:' num2str(x(end))]);
axis equal；
end

%%
function j = objective(x)
j = x(end)*10;
% + norm([x(N) - target(1), x(2*N) - target(2)]);
end

function [c,ceq] = dynamiccon(x, target, N, CFD, time0)
xs = x(1: N);
ys = x(N+1: N*2);
theta0 = x(2*N+1);
thetadots = x(2*N+2: 3*N);
dt = x(end);
mu = 0.8;
epsl = 0.15^2;
ceqx = zeros(N-1,1);
ceqy = zeros(N-1,1);
% ceqtheta = zeros(N-1,1);
% dxs = diff(xs);
% dys = diff(ys);
dthetas = thetadots*dt;
thetas = theta0 + [0; cumsum(dthetas)];
% linear sheer flow
% ceqx = xs(2:end) - xs(1:end-1) - cos(thetas(1:end-1)) - ys(1:end-1);
% ceqy = ys(2:end) - ys(1:end-1) - sin(thetas(1:end-1));
% global UUU VVV OOO XMIN XMAX YMIN YMAX cfd_framerate time_span cf

for i = 1:N-1
    time = i*dt + time0;
    pos = [xs(i);ys(i)];
%     posX = xs(i);
%     posY = ys(i);

%     [flowU, flowV, ~]= adapt_time_interp(CFD, time,posX,posY);
%     ceqx(i) = dxs(i) - (mu*cos(thetas(i)) + flowU)*dt;
%     ceqy(i) = dys(i) - (mu*sin(thetas(i)) + flowV)*dt;
%     opts = odeset('RelTol',1e-2,'AbsTol',1e-3);
%     [~,z] = ode113(@(t,z) odefcn(t,z,mu,time,CFD, thetas(i), thetadots(i)), [0 dt/2 dt], [xs(i); ys(i)],opts)
    f = @(t,z) odefcn(t,z,mu,time,CFD,thetas(i),thetadots(i));
    k1 = f(0, pos);
    k2 = f(dt/2, pos + dt*k1/2);
    k3 = f(dt/2, pos + dt*k2/2);
    k4 = f(dt, pos + dt*k3);
    pos = pos + dt*(k1 + 2*k2 + 2*k3 + k4)/6;

    ceqx(i) = xs(i+1) - pos(1);
    ceqy(i) = ys(i+1) - pos(2);

%     ceqtheta(i) = dthetas(i) - thetadots(i)*dt;
end
ceq = cat(1,ceqx, ceqy);
c = norm([x(N) - target(1), x(2*N) - target(2)]) - epsl;
end

function dzdt = odefcn(t,z,mu,time,CFD,theta,thetadot)
[flowU, flowV, ~]= adapt_time_interp(CFD,time+t,z(1),z(2));
dzdt = zeros(2,1);
dzdt(1) = mu*cos(theta+t*thetadot) + flowU;
dzdt(2) = mu*sin(theta+t*thetadot) + flowV;
end



function stop = optimplottraj(savename, x,optimValues,state,varargin)
% OPTIMPLOTTRAJ Plot current point at each iteration.
%
%   STOP = OPTIMPLOTTRAJ(X,OPTIMVALUES,STATE) plots the current point, X, as a
%   bar plot of its elements at the current iteration.
%
%   Example:
%   Create an options structure that will use OPTIMPLOTX
%   as the plot function
%       options = optimset('PlotFcns',@optimplottraj);
%
%   Pass the options into an optimization problem to view the plot
%       fminbnd(@sin,3,10,options)
stop = false;

% Reshape if x is a matrix
N = (length(x)-1)/3;
if optimValues.iteration == 0
    % The 'iter' case is  called during the zeroth iteration,
    % but it now has values that were empty during the 'init' case
    plottraj = plot(x(1:N),x(N+1:2*N),'.'); hold on;
    plot(x(1:N),x(N+1:2*N),'k--');
    plotstart = plot(x(1),x(N+1),'r*');
    plotend = plot(x(N),x(2*N),'b*');
    title('Current trajectory');
    ylabel('y');
    xlabel('x');
    set(gca,'xlim',[-24,0])
    set(gca,'ylim',[-5,5])
    set(gca,'DataAspectRatio',[1 1 1])

    set(plottraj,'Tag','optimplottraj');
    set(plotstart,'Tag','optimplotstart');
    set(plotend,'Tag','optimplotend');
    xtraj = x;
    save([savename 'optimtraj.mat'],'xtraj');
    %     set(gcf,'Position',[179 163 2283 1092]);
    %     subplot(5,1,[1 3],gca);
else
    %     q = findobj(get(gcf,'Children'),'Type','Axes');
    %     axesN = length(q);
    %     subplot(axesN+2,1,[1 3],gca);
    %     k = 4;
    %     for i = 1:axesN
    %         if q(i) ~= gca
    %             subplot(axesN+2,1,k,q(i));
    %             k = k + 1;
    %         end
    %     end
    plottraj = findobj(get(gca,'Children'),'Tag','optimplottraj');
    plotstart = findobj(get(gca,'Children'),'Tag','optimplotstart');
    plotend = findobj(get(gca,'Children'),'Tag','optimplotend');
    set(plottraj,'Xdata',x(1:N));
    set(plottraj,'Ydata',x(N+1:2*N));
    set(plotstart,'Xdata',x(1));
    set(plotstart,'Ydata',x(N+1));
    set(plotend,'Xdata',x(N));
    set(plotend,'Ydata',x(2*N));
    load([savename 'optimtraj.mat'],'xtraj');
    xtraj = [xtraj x];
    save([savename 'optimtraj.mat'],"xtraj");
end
end
%% functions to do type conversions
function x = ast(x)
x = x.astype('float64');
end
function x = ListsToCells(x)
x = cell(x);
x = cellfun(@cell ,x, 'UniformOutput', false);
end
function x = AllCellsToDouble(x)
x = ListsToCells(x);
for i = 1: length(x)
    x{i} = cellfun(@(y) ast(y), x{i},'UniformOutput',false);
    x{i} = cellfun(@(y) double(y), x{i},'UniformOutput',false);
end
% x = cellfun(@(x) cellfun(@(y) double(ast(y)), x,'UniformOutput',false), x,'UniformOutput',false);
end
%%  load data with python function
function [UUU, VVV, OOO, XMIN, XMAX, YMIN, YMAX, cfd_framerate, time_span, cf] = load_data(time_span)
% pe = pyenv('Version','/home/yusheng/anaconda3/bin/python3',"ExecutionMode","OutOfProcess");
% pe = pyenv('Version','/home/yusheng/anaconda3/bin/python3');
% pyrun("import sys");
% pyrun("sys.path.append('/home/yusheng/navigation_envs/dipole_new')");

% global UUU VVV OOO XMIN XMAX YMIN YMAX cfd_framerate time_span cf
pth = py.sys.path;
pth.append('/home/yusheng/navigation_envs/dipole_new');

cf = py.importlib.import_module('CFDfunctions');
source_path = '/home/yusheng/CFDadapt/np/';
level_limit = 3;
flow_data = cf.adapt_load_data(time_span,  source_path, level_limit);
cfd_framerate = flow_data{1};
time_span = flow_data{2};
UUU = flow_data{3};
VVV = flow_data{4};
OOO = flow_data{5};
XMIN = flow_data{6};
XMAX = flow_data{7};
YMIN = flow_data{8};
YMAX = flow_data{9};

% alternative way to call this python function...
% [cfd_framerate,time_span,UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX] = ...
%     pyrun("[cfd_framerate,time_span,UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX] = cf.adapt_load_data(a,b,c)",...
%     ["cfd_framerate","time_span","UUU","VVV","OOO","XMIN","XMAX","YMIN","YMAX"],...
%     a = time_span, b = source_path, c = level_limit);
end
