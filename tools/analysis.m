%%
load('/home/yusheng/smarties/apps/dipole_adapt/paper/egoLRGrad1/grading_results.mat');
obs_x_ego = reshape(observation(1,:,:),1,[]);
obs_y_ego = reshape(observation(2,:,:),1,[]);
obs_u_ego = reshape(observation(3,:,:),1,[]);
obs_v_ego = reshape(observation(4,:,:),1,[]);
obs_du_ego = reshape(observation(5,:,:),1,[]);
obs_dv_ego = reshape(observation(6,:,:),1,[]);
valid = (obs_x_ego~=0);
obs_x_ego = obs_x_ego(valid);
obs_y_ego = obs_y_ego(valid);
obs_u_ego = obs_u_ego(valid);
obs_v_ego = obs_v_ego(valid);
obs_du_ego = obs_du_ego(valid);
obs_dv_ego = obs_dv_ego(valid);
figure,subplot(6,1,1),histogram(obs_x_ego); ylabel('$\Delta x_b$')
subplot(6,1,2),histogram(obs_y_ego); ylabel('$\Delta y_b$')
subplot(6,1,3),histogram(obs_u_ego); ylabel('$u_b$')
subplot(6,1,4),histogram(obs_v_ego); ylabel('$v_b$')
subplot(6,1,5),histogram(obs_du_ego); ylabel('$\partial u/\partial y_b$')
subplot(6,1,6),histogram(obs_dv_ego); ylabel('$\partial v/\partial y_b$')

%%
load('/home/yusheng/smarties/apps/dipole_adapt/paper/geo8/grading_results.mat');
obs_x_geo = reshape(observation(1,:,:),1,[]);
obs_y_geo = reshape(observation(2,:,:),1,[]);
obs_u_geo = reshape(observation(3,:,:),1,[]);
obs_v_geo = reshape(observation(4,:,:),1,[]);
obs_theta_geo = reshape(observation(5,:,:),1,[]);
valid = (obs_x_geo~=0);
obs_x_geo = obs_x_geo(valid);
obs_y_geo = obs_y_geo(valid);
obs_u_geo = obs_u_geo(valid);
obs_v_geo = obs_v_geo(valid);
obs_theta_geo = obs_theta_geo(valid);

figure,subplot(5,1,1),histogram(obs_x_geo); ylabel('$\Delta x$')
subplot(5,1,2),histogram(obs_y_geo); ylabel('$\Delta y$')
subplot(5,1,3),histogram(obs_u_geo); ylabel('$\theta$')
subplot(5,1,4),histogram(obs_v_geo); ylabel('$u$')
subplot(5,1,5),histogram(obs_theta_geo); ylabel('$v$')