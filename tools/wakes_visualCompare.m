close all;
clear;
%%
figureDefaultSettings;

%% reduced only
close all;
v = VideoWriter('reduced_wake');
v.FrameRate = 20;
open(v);
figure('Position',[955 759 764 539]);
lam = 4;
A = 0.3;
Gamma = 3;
U = Gamma/2/lam*tanh(2*pi*A/lam) - 1;
phase = mod(0,lam);
vortexNum = ceil(24/lam);
vortexUpX = mod(phase+24,lam)-24 + (0:vortexNum-1)*lam;
vortex_up = plot(vortexUpX,A*ones(size(vortexUpX)),'r.','MarkerSize',24); hold on
vortexDownX = mod(phase+24+lam/2,lam)-24+(0:vortexNum-1)*lam;
vortex_down = plot(vortexDownX,-A*ones(size(vortexDownX)),'b.','MarkerSize',24);

axis equal;
hold on
axis off
xlim([-23.5,0]);
ylim([-6,6]);

frame = getframe(gcf);
writeVideo(v,frame);
for time = 1/20:1/20:9
    phase = mod(U*time,lam);
    vortexUpX = mod(phase+24,lam)-24 + (0:vortexNum-1)*lam;
    vortex_up.XData = vortexUpX;
    vortexDownX = mod(phase+24+lam/2,lam)-24+(0:vortexNum-1)*lam;
    vortex_down.XData = vortexDownX;
    drawnow;
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v)

%% together
close all;
v = VideoWriter('CFD_reduced_wake_comparison');
v.FrameRate = 20;
open(v);
figure('Position',[319 403 764 919]);
lam = 4;
A = 0.3;
Gamma = 3;
U = Gamma/2/lam*tanh(2*pi*A/lam) - 1;

subplot(2,1,1);
[bg,map] = imread(['/home/yusheng/CFDadapt/Movie/movie' num2str(0,'%04.f') '.png'],"png");
show_image = image([-24,8],[8,-8],flipud(bg));

hold on
xlim([-23.5,0]);
ylim([-6,6]);
axis equal;
axis off
subplot(2,1,2);
phase = mod(0,lam);
vortexNum = ceil(24/lam);
vortexUpX = mod(phase+24,lam)-24 + (0:vortexNum-1)*lam;
vortex_up = plot(vortexUpX,A*ones(size(vortexUpX)),'r.','MarkerSize',24); hold on
vortexDownX = mod(phase+24+lam/2,lam)-24+(0:vortexNum-1)*lam;
vortex_down = plot(vortexDownX,-A*ones(size(vortexDownX)),'b.','MarkerSize',24);
xlim([-23.5,0]);
ylim([-6,6]);
axis equal;
axis off
frame = getframe(gcf);
writeVideo(v,frame);
for time = 1/20:1/20:9
    [bg,map] = imread(['/home/yusheng/CFDadapt/Movie/movie' num2str(mod(time,4.5)*20,'%04.f') '.png'],"png");hold on;
    show_image.CData = flipud(bg);
    phase = mod(U*time,lam);
    vortexUpX = mod(phase+24,lam)-24 + (0:vortexNum-1)*lam;
    vortex_up.XData = vortexUpX;
    vortexDownX = mod(phase+24+lam/2,lam)-24+(0:vortexNum-1)*lam;
    vortex_down.XData = vortexDownX;
    drawnow;
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v)

%% overlay
close all;
clear;
v = VideoWriter('CFD_reduced_wake_comparison_overlay');
v.FrameRate = 20;
open(v);
figure('Position',[955 759 764 539]);
lam = 4;
A = 0.3;
Gamma = 3;
U = Gamma/2/lam*tanh(2*pi*A/lam) - 1;

[bg,map] = imread(['/home/yusheng/CFDadapt/Movie/movie' num2str(0,'%04.f') '.png'],"png");
show_image = image([-24,8],[8,-8],flipud(bg));

hold on

axis equal;
xlim([-23.5,0]);
ylim([-6,6]);
axis off
phase = mod(0,lam);
vortexNum = ceil(24/lam);
vortexUpX = mod(phase+24,lam)-24 + (0:vortexNum-1)*lam;
vortex_up = plot(vortexUpX,A*ones(size(vortexUpX)),'r.','MarkerSize',24);
vortexDownX = mod(phase+24+lam/2,lam)-24+(0:vortexNum-1)*lam;
vortex_down = plot(vortexDownX,-A*ones(size(vortexDownX)),'b.','MarkerSize',24);
frame = getframe(gcf);
writeVideo(v,frame);
for time = 1/20:1/20:9
    [bg,map] = imread(['/home/yusheng/CFDadapt/Movie/movie' num2str(mod(time,4.5)*20,'%04.f') '.png'],"png");hold on;
    show_image.CData = flipud(bg);
    phase = mod(U*time,lam);
    vortexUpX = mod(phase+24,lam)-24 + (0:vortexNum-1)*lam;
    vortex_up.XData = vortexUpX;
    vortexDownX = mod(phase+24+lam/2,lam)-24+(0:vortexNum-1)*lam;
    vortex_down.XData = vortexDownX;
    drawnow;
%     disp(time)
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v);