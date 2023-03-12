function obs = observation_reduced_geo(states, time, target)
states = states(:);
target = target(:);
[flowU, flowV, vorticity]= adapt_time_interp(CFD,time,states(1),states(2));



obs = [target-states(1:2); states(3); flowU; flowV];
end