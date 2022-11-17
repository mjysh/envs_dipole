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
time = 0;
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
[gux,guy] = gradient(flowU_reconstructed,x(2)-x(1));
[gvx,gvy] = gradient(flowV_reconstructed,x(2)-x(1));
[gsx,gsy] = gradient(sqrt(flowV_reconstructed.^2+flowU_reconstructed.^2),x(2)-x(1));
figure,quiver(X(1:5:end,1:5:end),Y(1:5:end,1:5:end),gux(1:5:end,1:5:end)./sqrt(gux(1:5:end,1:5:end).^2+guy(1:5:end,1:5:end).^2),guy(1:5:end,1:5:end)./sqrt(gux(1:5:end,1:5:end).^2+guy(1:5:end,1:5:end).^2));

hold on, quiver(X(1:5:end,1:5:end),Y(1:5:end,1:5:end),gvx(1:5:end,1:5:end)./sqrt(gvx(1:5:end,1:5:end).^2+gvy(1:5:end,1:5:end).^2),gvy(1:5:end,1:5:end)./sqrt(gvx(1:5:end,1:5:end).^2+gvy(1:5:end,1:5:end).^2));
axis equal
figure,quiver(X(1:5:end,1:5:end),Y(1:5:end,1:5:end),gsx(1:5:end,1:5:end)./sqrt(gsx(1:5:end,1:5:end).^2+gsy(1:5:end,1:5:end).^2),gsy(1:5:end,1:5:end)./sqrt(gsx(1:5:end,1:5:end).^2+gsy(1:5:end,1:5:end).^2));
axis equal
%%
figure;
cav = curl(flowU_reconstructed,flowV_reconstructed);
pcolor(X,Y,cav);shading interp
hold on, quiver(X(1:5:end,1:5:end),Y(1:5:end,1:5:end),flowU_reconstructed(1:5:end,1:5:end),flowV_reconstructed(1:5:end,1:5:end));
axis equal
contour(X,Y,flowU_reconstructed,[-1,-0.8,-0.6,0]);
colorbar;