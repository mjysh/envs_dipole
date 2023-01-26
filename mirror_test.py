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
        return np.array([action])

def test(args):
    ############## Hyperparameters ##############
    
    policy_path = args.policy_path
    N1 = 128
    N2 = 128
    with open(policy_path+"/paramToUse.txt","r") as paramFile:
        settings = paramFile.readline().strip()
    env = fish.DipoleSingleEnv(paramSource = 'envParam_'+settings)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if args.swimmer_init:
        initPos = args.swimmer_init
    else:
        initPos = [-12.0,-2.15,0.0]
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
    env = wrappers.Monitor(env, './Movies/test',force = True)
    obs = env.reset(position = initPos, target = target_pos, init_time = None)
    env.done = False
    for t in range(max_timesteps):
        obs[0] = obs[0]
        obs[1] = -obs[1]
        obs[2] = obs[2]
        obs[3] = -obs[3]
        obs[4] = -obs[4]
        obs[5] = obs[5]
        action = trained_policy.obs_to_act(obs)
        action = -action
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
    savemat(policy_path+"/random_test_data.mat", mdic, oned_as='row', do_compression=True)
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
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, default='/home/yusheng/smarties/apps/dipole_adapt/paper/egoLRGrad1')
    parser.add_argument('--policy_num', type=int, default = 0)
    parser.add_argument('--target_pos', type=float, nargs=2)
    parser.add_argument('--swimmer_init', type=float, nargs=3)
    parser.add_argument('--max_timesteps', action="store", default=1000)
    args = parser.parse_args()
    test(args)