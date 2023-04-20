function obs = observation_reduced_geo(states, time, target)
states = states(:);
target = target(:);
[flowU,flowV] = reducedFlow(states,time);

obs = [target-states(1:2); states(3); flowU; flowV];
end
