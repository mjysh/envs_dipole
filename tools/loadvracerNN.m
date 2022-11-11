function vracerNN = loadvracerNN(path,Nobs,N1,N2,Naction)

ID = fopen([path '/agent_00_net_weights.raw']);
Params = fread(ID,'float');
ID = fopen([path '/agent_00_scaling.raw']);
SCALE = fread(ID,'double');
obs_mean = SCALE(1:Nobs);
obs_scale = SCALE(Nobs+1:2*Nobs);
pos = 1;
posNext = pos + N1*Nobs - 1;
W1 = reshape(Params(pos:posNext),N1,Nobs);
pos = posNext + 1;
posNext = pos + N1 - 1;
B1 = Params(pos:posNext);
pos = posNext + 1;
posNext = pos + N2*N1 - 1;
W2 = reshape(Params(pos:posNext),N2,N1);
pos = posNext + 1;
posNext = pos + N2 - 1;
B2 = Params(pos:posNext);
pos = posNext + 1;
posNext = pos + N2 - 1;
W_res = Params(pos:posNext);
pos = posNext + 1;
posNext = pos + N2 - 1;
B_res = Params(pos:posNext);
pos = posNext + 1;
posNext = pos + N2*(Naction+1) - 1;
W3 = reshape(Params(pos:posNext),Naction+1,N2);
pos = posNext + 1;
posNext = pos + (Naction+1) - 1;
B3 = Params(pos:posNext);
pos = posNext + 1;
posNext = pos + Naction - 1;
STD_action = Params(pos:posNext);
assert(posNext == length(Params))
vracerNN = struct('obs_mean',obs_mean,'obs_scale',obs_scale,'W1',W1,'B1',B1,'W2',W2,'B2',B2,'W_res',W_res,'B_res',B_res,'W3',W3,'B3',B3,'std_action',STD_action);
end