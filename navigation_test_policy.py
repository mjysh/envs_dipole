#!/usr/bin/env python3
import sys
from PIL import Image
import torch
import numpy as np
import navigationAll_env as fish
import time
from gym import wrappers
device = torch.device("cpu")
class VRACER_NN():
    def __init__(self, path,Nobs,N1,N2,Naction):
        Params    = np.fromfile(path + "/agent_00_net_weights.raw", dtype=np.float32)
        SCALE    = np.fromfile(path + "/agent_00_scaling.raw", dtype=np.float64)
        self.obs_mean = SCALE[:Nobs].reshape((1,-1))
        self.obs_scale = SCALE[Nobs:2*Nobs].reshape((1,-1))

        pos = 0
        posNext = pos + N1*Nobs
        self.W1 = np.reshape(Params[pos:posNext],(Nobs,N1))
        
        pos = posNext
        posNext = pos + N1
        self.B1 = Params[pos:posNext].reshape((1,-1))
        
        pos = posNext
        posNext = pos + N2*N1
        self.W2 = np.reshape(Params[pos:posNext],(N1,N2))
        
        pos = posNext
        posNext = pos + N2
        self.B2 = Params[pos:posNext].reshape((1,-1))
        
        pos = posNext
        posNext = pos + N2
        self.W_res = Params[pos:posNext].reshape((1,-1))
        
        pos = posNext
        posNext = pos + N2
        self.B_res = Params[pos:posNext].reshape((1,-1))
        
        pos = posNext
        posNext = pos + N2*(Naction+1)
        self.W3 = np.reshape(Params[pos:posNext],(N2,Naction+1))
        
        pos = posNext
        posNext = pos + (Naction+1)
        self.B3 = Params[pos:posNext].reshape((1,-1))
        
        pos = posNext
        posNext = pos + Naction
        self.STD_action = Params[pos:posNext]
        assert(posNext == len(Params))
        print("Policy successfully loaded!\n")
    def obs_to_act(self, obs):
        obs = obs.reshape((1,-1))
        activation = np.tanh

        obs = (obs - self.obs_mean)*self.obs_scale
        s1 = obs@self.W1 + self.B1
        i1 = activation(s1)
        s2 = i1@self.W2 + self.B2
        i2 = activation(s2)

        s_res = self.W_res*i1 + self.B_res
        i_res = s_res + i2
        s3 = i_res@self.W3 + self.B3

        out = activation(s3).reshape((-1,))
        action = out[1]
        value = out[0]
        return [action]
def generalization_test(args):
    from tqdm import tqdm
    if args.test_bound:
        boundL, boundR, boundD, boundU = args.test_bound
    else:
        boundL = -23.0
        boundR = -1.0
        boundD = -5.0
        boundU = 5.0
    print(f"Test with initial position within left:{boundL}, right:{boundR}, bottom:{boundD}, top:{boundU}")
    #######pre-generated initial conditions############
    if args.vary_target:
        target_x, target_y, init_theta = [arr.flatten() for 
        arr in np.meshgrid(np.arange(boundL,boundR+0.5,0.5),
                           np.arange(boundD,boundU+0.5,0.5),
                           np.arange(0,2*np.pi,np.pi/18))]
        n_episodes = len(init_theta)
        if args.swimmer_init:
            init_x = np.repeat(args.swimmer_init[0],n_episodes)
            init_y = np.repeat(args.swimmer_init[1],n_episodes)
        else:
            init_x = np.repeat(-12,n_episodes)
            init_y = np.repeat(-2.15,n_episodes)
        print(f"Test with swimmer starting from {init_x[0],init_y[0]}")
    else:
        init_x, init_y, init_theta = [arr.flatten() for 
        arr in np.meshgrid(np.arange(boundL,boundR+0.5,0.5),
                           np.arange(boundD,boundU+0.5,0.5),
                           np.arange(0,2*np.pi,np.pi/18))]
        n_episodes = len(init_x)
        if args.target_pos:
            target_x = np.repeat(args.target_pos[0],n_episodes)
            target_y = np.repeat(args.target_pos[1],n_episodes)
        else:
            target_x = np.repeat(-12,n_episodes)
            target_y = np.repeat(2.15,n_episodes)
        print(f"Test with target at {args.target_pos}")
    ############## Hyperparameters ##############
    if args.policy_num:
        policy_paths = [args.policy_path+str(k) for k in range(1,args.policy_num+1)]
    else:
        policy_paths = [args.policy_path]
    success_rate = []
    average_time = []
    average_success_time = []
    
    for k,policy_path in enumerate(policy_paths):
        print(policy_path)
        N1 = 128
        N2 = 128
                  # num of episodes to run
        render = False           # render the environment
        max_timesteps = 1500
        if args.env_setting:
            settings = args.env_setting
        else:
            with open(policy_path+"/paramToUse.txt","r") as paramFile:
                settings = paramFile.readline().strip()
        env = fish.DipoleSingleEnv(paramSource = 'envParam_'+settings)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        # set the length of an episode
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env, max_episode_steps=max_timesteps)
        trained_policy = VRACER_NN(path = policy_path, Nobs = obs_dim, N1 = N1, N2 = N2, Naction = action_dim)
        theta_history = np.zeros((max_timesteps,n_episodes))
        x_history = np.zeros((max_timesteps,n_episodes))
        y_history = np.zeros((max_timesteps,n_episodes))
        # obs_history = np.zeros((obs_dim,max_timesteps,n_episodes))
        action_history = np.zeros((max_timesteps,n_episodes))
        success = 0
        total_time = []
        total_success_time = []
        rewards = []
        for ep in tqdm(range(n_episodes)):
            ep_reward = 0
            obs = env.reset(position = [init_x[ep],init_y[ep],init_theta[ep]], 
                            target = [target_x[ep],target_y[ep]], init_time = 0)
            env.done = False
            for t in range(max_timesteps):
                theta_history[t,ep] = env.pos[-1]
                x_history[t,ep] = env.pos[0]
                y_history[t,ep] = env.pos[1]
                action = trained_policy.obs_to_act(obs)
                # obs_history[:,t,ep] = obs
                action_history[t,ep] = action[0]
                obs, reward, env.done, _ = env.step(action)
                ep_reward += reward
                if render:
                    env.render()
                    from pyglet.window import key                
                    @env.viewer.window.event
                    def on_key_press(symbol, modifiers):
                        print(symbol)
                        print(key.Q)
                        if symbol == key.Q:
                            env.done = True
                if env.done:
                    break
            rewards.append(ep_reward)
            if ep_reward > 100:
                success += 1
                total_success_time.append(t+1)
            total_time.append(t+1)
            ep_reward = 0
            env.close()
        success_rate.append(success/n_episodes)
        average_time.append(sum(total_time)/n_episodes)
        average_success_time.append(sum(total_success_time)/success if success else None)
        if args.vary_target:
            mdic = {"initX": init_x[0],
                "initY": init_y[0],
                "initTheta": init_theta,
                "targetX": target_x,
                "targetY": target_y,
                "reward": rewards,
                "totTime": total_time,
                "trajX": x_history,
                "trajY": y_history,
                "trajTheta": theta_history,
                # "observation": obs_history,
                "action": action_history
                }
            from scipy.io import savemat
            savemat(policy_path+f"/success_region_varytarget_{init_x[0]}_{init_y[0]}_{settings}.mat", mdic, oned_as='row', do_compression=True)
        else:
            mdic = {"initX": init_x,
                "initY": init_y,
                "initTheta": init_theta,
                "target": [target_x[0],target_y[0]],
                "reward": rewards,
                "totTime": total_time,
                "trajX": x_history,
                "trajY": y_history,
                "trajTheta": theta_history,
                # "observation": obs_history,
                "action": action_history
                }
            from scipy.io import savemat
            savemat(policy_path+f"/success_region{target_x[0]}_{target_y[0]}_{settings}.mat", mdic, oned_as='row', do_compression=True)
        print('total:', n_episodes, 'success:', success)
        print('success rate:', success/n_episodes)
    res_strs = []
    policy_index = 1
    for sr,at,ast in zip(success_rate,average_time,average_success_time):
        res_strs.append(f'policy {policy_index}: success rate:{sr}, average time:{at}, average success time:{ast}\n')
        print(res_strs[-1])
        policy_index += 1
    from datetime import date
    with open(args.policy_path+'_generalization_'+str(date.today())+'.txt','w') as resultFile:
        resultFile.writelines(res_strs)
def sourceseeking_test(args):
    from tqdm import tqdm
    #######pre-generated initial conditions############
    init_x, init_y, init_theta = [arr.flatten() for 
    arr in np.meshgrid(np.arange(-18.0,-6.0,0.5),np.arange(-3,3.5,0.5),np.arange(0,2*np.pi,np.pi/18))]
    # target_pos = [0, 0]

    ############## Hyperparameters ##############
    if args.policy_num:
        policy_paths = [args.policy_path+str(k) for k in range(1,args.policy_num+1)]
    else:
        policy_paths = [args.policy_path]
    success_rate = []
    average_time = []
    average_success_time = []
    
    for k,policy_path in enumerate(policy_paths):
        print(policy_path)
        N1 = 128
        N2 = 128
        n_episodes = len(init_x)          # num of episodes to run
        render = False           # render the environment
        max_timesteps = 1000
        if args.env_setting:
            settings = args.env_setting
        else:
            with open(policy_path+"/paramToUse.txt","r") as paramFile:
                settings = paramFile.readline().strip()
        assert 'Sourceseeking' in settings
        env = fish.DipoleSingleEnv(paramSource = 'envParam_'+settings)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        # set the length of an episode
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env, max_episode_steps=max_timesteps)
        trained_policy = VRACER_NN(path = policy_path, Nobs = obs_dim, N1 = N1, N2 = N2, Naction = action_dim)
        theta_history = np.zeros((max_timesteps,n_episodes))
        x_history = np.zeros((max_timesteps,n_episodes))
        y_history = np.zeros((max_timesteps,n_episodes))
        # obs_history = np.zeros((obs_dim,max_timesteps,n_episodes))
        action_history = np.zeros((max_timesteps,n_episodes))
        success = 0
        total_time = []
        total_success_time = []
        rewards = []
        print(f'Testing for going above -6')
        for ep in tqdm(range(n_episodes)):
            ep_reward = 0
            obs = env.reset(position = [init_x[ep],init_y[ep],init_theta[ep]], init_time = 0)
            env.done = False
            for t in range(max_timesteps):
                theta_history[t,ep] = env.pos[-1]
                x_history[t,ep] = env.pos[0]
                y_history[t,ep] = env.pos[1]
                action = trained_policy.obs_to_act(obs)
                # obs_history[:,t,ep] = obs
                action_history[t,ep] = action[0]
                obs, reward, env.done, _ = env.step(action)
                ep_reward += reward
                if render:
                    env.render()
                    from pyglet.window import key                
                    @env.viewer.window.event
                    def on_key_press(symbol, modifiers):
                        print(symbol)
                        print(key.Q)
                        if symbol == key.Q:
                            env.done = True
                if env.done:
                    break
            rewards.append(ep_reward)
            if ep_reward > 100:
                success += 1
                total_success_time.append(t+1)
            total_time.append(t+1)
            ep_reward = 0
            env.close()
        success_rate.append(success/n_episodes)
        average_time.append(sum(total_time)/n_episodes)
        average_success_time.append(sum(total_success_time)/success if success else None)
        mdic = {"initX": init_x,
            "initY": init_y,
            "initTheta": init_theta,
            "threshold": -8,
            "reward": rewards,
            "totTime": total_time,
            "trajX": x_history,
            "trajY": y_history,
            "trajTheta": theta_history,
            # "observation": obs_history,
            "action": action_history
            }
        from scipy.io import savemat
        savemat(policy_path+f"/success_region_sourceseeking_{settings}-8.mat", mdic, oned_as='row', do_compression=True)
        print('total:', n_episodes, 'success:', success)
        print('success rate:', success/n_episodes)
    res_strs = []
    policy_index = 1
    for sr,at,ast in zip(success_rate,average_time,average_success_time):
        res_strs.append(f'policy {policy_index}: success rate:{sr}, average time:{at}, average success time:{ast}\n')
        print(res_strs[-1])
        policy_index += 1
    from datetime import date
    with open(args.policy_path+'_generalization_'+str(date.today())+'.txt','w') as resultFile:
        resultFile.writelines(res_strs)
def grade_naive_policy(args):
    #######pre-generated initial conditions############
    init_swimmer = np.load('./swimmer_initpositions.npy')
    init_time = np.load('./init_time.npy')
    init_target = np.load('./target_positions.npy')
    assert len(init_time) == init_swimmer.shape[0]
    assert len(init_time) == init_target.shape[0]
    ############## Hyperparameters ##############
    # parent_path = args.policy_path.rsplit('/',1)[0]

    n_episodes = len(init_time)          # num of episodes to run
    render = False           # render the environment
    max_timesteps = args.max_timesteps
    policy_path = args.policy_path
    if args.env_setting:
        settings = args.env_setting
    else:
        with open(policy_path+"/paramToUse.txt","r") as paramFile:
            settings = paramFile.readline().strip()
    env = fish.DipoleSingleEnv(paramSource = 'envParam_'+settings)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # set the length of an episode
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=max_timesteps)
    theta_history = np.zeros((max_timesteps,n_episodes))
    x_history = np.zeros((max_timesteps,n_episodes))
    y_history = np.zeros((max_timesteps,n_episodes))
    obs_history = np.zeros((obs_dim,max_timesteps,n_episodes))
    action_history = np.zeros((max_timesteps,n_episodes))
    success = 0
    total_time = []
    total_success_time = []
    rewards = []
    for ep in range(n_episodes):
        ep_reward = 0
        # print(init_swimmer[ep,:])
        # print(init_target[ep,:])
        # print(init_time[ep])
        obs = env.reset(position = init_swimmer[ep,:], target = init_target[ep,:], init_time = init_time[ep])
        env.done = False
        for t in range(max_timesteps):
            theta_history[t,ep] = env.pos[-1]
            x_history[t,ep] = env.pos[0]
            y_history[t,ep] = env.pos[1]
            dx = env.target[0] - env.pos[0]
            dy = env.target[1] - env.pos[1]
            target_angle = np.arctan2(dy,dx)
            relort = angle_normalize(target_angle - env.pos[-1])

            action = [np.sign(relort)]
            obs_history[:,t,ep] = obs
            action_history[t,ep] = action[0]
            obs, reward, env.done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
                from pyglet.window import key                
                @env.viewer.window.event
                def on_key_press(symbol, modifiers):
                    print(symbol)
                    print(key.Q)
                    if symbol == key.Q:
                        env.done = True
            if env.done:
                break
        print('Episode: {}\tReward: {}'.format(ep, (ep_reward)))
        rewards.append(ep_reward)
        if ep_reward > 100:
            success += 1
            total_success_time.append(t+1)
        total_time.append(t+1)
        ep_reward = 0
        env.close()

    mdic = {
        "reward": rewards,
        "totTime": total_time,
        "trajX": x_history,
        "trajY": y_history,
        "trajTheta": theta_history,
        "action": action_history
        }
    from scipy.io import savemat
    savemat("./grading_naive_"+ settings+".mat", mdic, oned_as='row', do_compression=True)
    res_strs = []
    res_strs.append(f'naive policy: success rate:{success/n_episodes}, average time:{sum(total_time)/n_episodes}, average success time:{sum(total_success_time)/success if success else None}\n')
    print(res_strs[-1])
    from datetime import date
    with open('naive_grade_'+settings+str(date.today())+'.txt','w') as resultFile:
        resultFile.writelines(res_strs)
def grade_policy(args):
    #######pre-generated initial conditions############
    init_swimmer = np.load('./swimmer_initpositions.npy')
    init_time = np.load('./init_time.npy')
    init_target = np.load('./target_positions.npy')
    assert len(init_time) == init_swimmer.shape[0]
    assert len(init_time) == init_target.shape[0]
    ############## Hyperparameters ##############
    # parent_path = args.policy_path.rsplit('/',1)[0]
    if args.policy_num:
        policy_paths = [args.policy_path+str(k) for k in range(1,args.policy_num+1)]
    else:
        policy_paths = [args.policy_path]
    success_rate = []
    average_time = []
    average_success_time = []
    
    for k,policy_path in enumerate(policy_paths):
        print(policy_path)
        N1 = 128
        N2 = 128
        n_episodes = len(init_time)          # num of episodes to run
        render = False           # render the environment
        max_timesteps = args.max_timesteps
        if args.env_setting:
            settings = args.env_setting
        else:
            with open(policy_path+"/paramToUse.txt","r") as paramFile:
                settings = paramFile.readline().strip()
        env = fish.DipoleSingleEnv(paramSource = 'envParam_'+settings)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        # set the length of an episode
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env, max_episode_steps=max_timesteps)
        trained_policy = VRACER_NN(path = policy_path, Nobs = obs_dim, N1 = N1, N2 = N2, Naction = action_dim)
        theta_history = np.zeros((max_timesteps,n_episodes))
        x_history = np.zeros((max_timesteps,n_episodes))
        y_history = np.zeros((max_timesteps,n_episodes))
        obs_history = np.zeros((obs_dim,max_timesteps,n_episodes))
        action_history = np.zeros((max_timesteps,n_episodes))
        success = 0
        total_time = []
        total_success_time = []
        rewards = []
        for ep in range(n_episodes):
            ep_reward = 0
            # print(init_swimmer[ep,:])
            # print(init_target[ep,:])
            # print(init_time[ep])
            obs = env.reset(position = init_swimmer[ep,:], target = init_target[ep,:], init_time = init_time[ep])
            env.done = False
            for t in range(max_timesteps):
                theta_history[t,ep] = env.pos[-1]
                x_history[t,ep] = env.pos[0]
                y_history[t,ep] = env.pos[1]
                action = trained_policy.obs_to_act(obs)
                obs_history[:,t,ep] = obs
                action_history[t,ep] = action[0]
                obs, reward, env.done, _ = env.step(action)
                ep_reward += reward
                if render:
                    env.render()
                    from pyglet.window import key                
                    @env.viewer.window.event
                    def on_key_press(symbol, modifiers):
                        print(symbol)
                        print(key.Q)
                        if symbol == key.Q:
                            env.done = True
                if env.done:
                    break
            print('Episode: {}\tReward: {}'.format(ep, (ep_reward)))
            rewards.append(ep_reward)
            if ep_reward > 100:
                success += 1
                total_success_time.append(t+1)
            total_time.append(t+1)
            ep_reward = 0
            env.close()
        success_rate.append(success/n_episodes)
        average_time.append(sum(total_time)/n_episodes)
        average_success_time.append(sum(total_success_time)/success if success else None)
        mdic = {
            "reward": rewards,
            "totTime": total_time,
            "trajX": x_history,
            "trajY": y_history,
            "trajTheta": theta_history,
            "target":init_target
            }
        from scipy.io import savemat
        savemat(policy_path+f"/grading_results_{settings}.mat", mdic, oned_as='row', do_compression=True)
    res_strs = ['policy, success rate, average time, average success time\n']
    policy_index = 1
    for sr,at,ast in zip(success_rate,average_time,average_success_time):
        res_strs.append(f'{policy_index}, {sr}, {at}, {ast}\n')
        print(res_strs[-1])
        policy_index += 1
    from datetime import date
    with open(args.policy_path+f'_grade_{settings}' +str(date.today())+'.txt','w') as resultFile:
        resultFile.writelines(res_strs)
def test(args):
    ############## Hyperparameters ##############
    
    policy_path = args.policy_path
    N1 = 128
    N2 = 128
    if args.env_setting:
        settings = args.env_setting
    else:
        with open(policy_path+"/paramToUse.txt","r") as paramFile:
            settings = paramFile.readline().strip()
    env = fish.DipoleSingleEnv(paramSource = 'envParam_'+settings)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if args.swimmer_init:
        initPos = args.swimmer_init
    else:
        initPos = None
    if args.target_pos:
        target_pos = args.target_pos
    else:
        target_pos = [-12, 2.15]

    n_episodes = 1          # num of episodes to run
    max_timesteps = args.max_timesteps    # max timesteps in one episode
    render = True           # render the environment
    save_gif = False        # png images are saved in gif folder
    
    
    trained_policy = VRACER_NN(path = policy_path, Nobs = obs_dim, N1 = N1, N2 = N2, Naction = action_dim)
    pos_history = np.zeros((3,max_timesteps+1))
    obs_history = np.zeros((obs_dim,max_timesteps))
    action_history = np.zeros((max_timesteps))
    ep_reward = 0
    env = wrappers.Monitor(env, './Movies/test_single',force = True)
    obs = env.reset(position = initPos, target = target_pos, init_time = None)
    initPos = env.pos
    env.done = False
    for t in range(max_timesteps):
        action = trained_policy.obs_to_act(obs)
        pos_history[:,t] = env.pos
        obs_history[:,t] = obs
        action_history[t] = action[0]
        obs, reward, env.done, _ = env.step(action)
        ep_reward += reward
        # print(obs)
        if render:
            env.render()
            from pyglet.window import key                
            @env.viewer.window.event
            def on_key_press(symbol, modifiers):
                print(symbol)
                print(key.Q)
                if symbol == key.Q:
                    env.done = True
        if save_gif:
                img = env.render(mode = 'rgb_array')
                img = Image.fromarray(img)
                img.save('./gif/{}.jpg'.format(t))  
        if env.done:
            break
    mdic = {"initPos": initPos,
            "reward": ep_reward,
            "totTime": t+1,
            "trajectory": pos_history,
            "observation": obs_history,
            "target": target_pos,
            "action": action_history
            }
    from scipy.io import savemat
    savemat(policy_path+f"/random_test_data_{settings}.mat", mdic, oned_as='row', do_compression=True)
#        import matplotlib.pyplot as plt
#        from matplotlib import rc
#        import matplotlib as mpl
        
#        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#        ## for Palatino and other serif fonts use:
#        #rc('font',**{'family':'serif','serif':['Palatino']})
#        rc('text', usetex=True)
#        fig,(ax1,ax2,ax3) = plt.subplots(3,1)
#        T = np.linspace(0,20,121)
#        ax1.plot(T[0],obs_his[0,0],'ro',T,obs_his[:,0],'k')
#        plt.show()
#        ax1.set_ylabel('orientation',size = 14)
#        ax1.set_title('Trained Policy',size = 20)
#        
#        
#        ax2.plot(T[0],obs_his[0,2],'ro',T,obs_his[:,2],'k')
#        plt.show()
#        ax2.set_ylabel('head angle',size = 14)
#        
#        ax3.plot(T[0],obs_his[0,1],'ro',T,obs_his[:,1],'k')
#        plt.show()
#        ax3.set_ylabel('tail angle',size = 14)
#        ax3.set_xlabel('time',fontsize = 14)
#        
#        fig,ax = plt.subplots()
#        
#        ax.plot(obs_his[0,2],obs_his[0,1],'ro',obs_his[:,2],obs_his[:,1],'k')
#        plt.axis([-np.pi, np.pi,-np.pi, np.pi])
#        ax.set_aspect('equal')
#        plt.show()
#        ax.set_xlabel('head angle',size = 14)
#        ax.set_ylabel('tail angle',size = 14)
#        ax.set_title('Trained Policy',size = 20)
    print('Episode: {}\tReward: {}'.format(1, (ep_reward)))
    ep_reward = 0
    env.close()
def angle_normalize(x,center = 0,half_period = np.pi):
    return (((x+half_period-center) % (2*half_period)) - half_period+center)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str ,choices=['policy_grading', 'generalization', 'single','naive_grading','sourceseeking_generalization'], required=True)
    
    parser.add_argument('--policy_path', type=str, default='tools/bestpolicy_geo')
    parser.add_argument('--env_setting', type=str, default='')
    parser.add_argument('--policy_num', type=int, default = 0)
    parser.add_argument('--target_pos', type=float, nargs=2)
    parser.add_argument('--swimmer_init', type=float, nargs=3)
    parser.add_argument('--max_timesteps', action="store", default=1000)
    parser.add_argument('--test_bound', type=float, nargs=4, default = [])
    parser.add_argument('--vary_target', action="store_true", default = False)
    args = parser.parse_args()
    print(f'MODE: {args.test_mode}')
    if args.test_mode == 'policy_grading':
        grade_policy(args)
    elif args.test_mode == 'generalization':
        generalization_test(args)
    elif args.test_mode == 'single':
        test(args)
    elif args.test_mode == 'naive_grading':
        grade_naive_policy(args)
    elif args.test_mode == 'sourceseeking_generalization':
        sourceseeking_test(args)