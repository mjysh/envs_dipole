function [n_angle, T, targetX, targetY, initX, initY, initTheta, reward] = a(initX, initY, initTheta, reward)
n_angle = 36;
dir = './';
T = readlines([dir 'success_region.txt']);
targetStrings = split(T(1));
targetX = str2double(targetStrings(3));
targetY = str2double(targetStrings(4));

initX = zeros(length(T)-2,1);
initY = zeros(length(T)-2,1);
initTheta = zeros(length(T)-2,1);
reward = zeros(length(T)-2,1);
end