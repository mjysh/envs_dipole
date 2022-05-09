function [action,value] = obs_to_act(obs, vracerNN)
obs = obs(:);
activation = @tanh;

obs = (obs-vracerNN.obs_mean).*vracerNN.obs_scale;
s1 = vracerNN.W1*obs + vracerNN.B1;
i1 = activation(s1);

s2 = vracerNN.W2*i1 + vracerNN.B2;
i2 = activation(s2);


s_res = vracerNN.W_res.*i1 + vracerNN.B_res;
i_res = s_res + i2;

s3 = vracerNN.W3*i_res + vracerNN.B3;
out = tanh(s3);
action = out(2);
value = out(1);
end