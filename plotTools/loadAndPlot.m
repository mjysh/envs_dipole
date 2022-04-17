clear;
close all;


load("lab12-12optimtraj.mat");
N = (height(xtraj)-1)/3;
plot(xtraj(1:N,1),xtraj(N+1:2*N,1));
hold on;
plot(xtraj(1:N,end),xtraj(N+1:2*N,end));
title([num2str(xtraj(end,1)) 'to' num2str(xtraj(end,end))]);