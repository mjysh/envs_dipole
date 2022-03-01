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

options = optimoptions("fmincon");
fun = @objective;  

%% 
N = 10;
lb = [ones(N,1)*(-Inf) ones(N,1)*(-5) ones(N,1)*(-pi)];
ub = -lb;  

A = [];
b = [];
Aeq = zeros(4,3*N);
Aeq(1,1) = 1;
Aeq(2,1+N) = 1;
Aeq(3,end) = 1;
Aeq(4,2*N) = 1;
beq = [0; 0; 0; 0];  

x0 = [zeros(N,2) zeros(N,1)];


nonlcon = @dynamiccon;
[x,fval,exitflag,output] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon);
plot(x(:,1),x(:,2),'*-');
axis equal;


%%
pe = pyenv('Version','/home/yusheng/anaconda3/bin/python3');

%%
function j = objective(x)
j = -x(end,1);
end
function [c,ceq] = dynamiccon(x)
xs = x(:,1);
ys = x(:,2);
thetas = x(:,3);
ceqx = xs(2:end) - xs(1:end-1) - cos(thetas(1:end-1)) - ys(1:end-1);
ceqy = ys(2:end) - ys(1:end-1) - sin(thetas(1:end-1));
ceq = cat(1,ceqx, ceqy);
c = [];
end