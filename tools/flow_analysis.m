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
CFD = load('CFDData.mat');
%%
time = 1;
x = linspace(-20,5,251);
y = linspace(-5,5,101);
[X,Y] = meshgrid(x,y);
flowU_reconstructed = zeros(size(X));
flowV_reconstructed = zeros(size(X));

%%
for ix = 1:length(x)
    for iy = 1:length(y)
        posX = X(iy,ix);
        posY = Y(iy,ix);
        [U_interp,V_interp,O_interp] = adapt_time_interp(CFD,time,posX, posY);
        flowU_reconstructed(iy,ix) = U_interp;
        flowV_reconstructed(iy,ix) = V_interp;
    end
end
%%
figure;
cav = curl(flowU_reconstructed,flowV_reconstructed);
pcolor(X,Y,cav);shading interp
hold on, quiver(X,Y,flowU_reconstructed,flowV_reconstructed);
axis equal
contour(X,Y,flowU_reconstructed,[-1,-0.8,-0.6,0]);
colorbar;