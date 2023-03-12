function obs = observation_reduced_ego(states, time, target)
states = states(:);
target = target(:);
ort = states(3);

R = [cos(ort), sin(ort);
    -sin(ort), cos(ort)];

left = states(1:2) + [-0.05*sin(ort); 0.05*cos(ort)];
right = states(1:2)*2 - left;

[flowUL, flowVL, vorticityL]= adapt_time_interp(CFD,time,left(1),left(2));
[flowUR, flowVR, vorticityR]= adapt_time_interp(CFD,time,right(1),right(2));
% [flowU, flowV, ~]= adapt_time_interp(CFD,time,states(1),states(2));
relPos = R*(target - states(1:2));
relFlowL = R*[flowUL; flowVL];
relFlowR = R*[flowUR; flowVR];
% disp(((flowVL+flowVR)/2 - flowV)/flowV)
% disp(((flowUL+flowUR)/2 - flowU)/flowU)
obs = [relPos; (relFlowL+relFlowR)/2; (relFlowL - relFlowR)/0.1];
end