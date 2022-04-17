function dado = backward_matrix(obs, vracerNN)

obs = (obs-vracerNN.obs_mean).*vracerNN.obs_scale;
k1 = vracerNN.W1*obs + vracerNN.B1;
i1 = tanh(k1);

k2 = vracerNN.W2*i1 + vracerNN.B2;
i2 = tanh(k2);


k_res = vracerNN.W_res.*i1 + vracerNN.B_res;
i_res = k_res + i2;

k3 = vracerNN.W3*i_res + vracerNN.B3;
a = tanh(k3);

dado = diag(1-a.^2)*vracerNN.W3 * (diag(vracerNN.W_res) + diag(1-i2.^2)*vracerNN.W2) * diag(1-i1.^2)*vracerNN.W1*diag(vracerNN.obs_scale);
end