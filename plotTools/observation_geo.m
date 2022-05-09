function obs = observation_geo(states, time, CFD, target)
states = states(:);
target = target(:);
[flowU, flowV, vorticity]= adapt_time_interp(CFD,time,states(1),states(2));

obs = [target-states(1:2); states(3); flowU; flowV];
end