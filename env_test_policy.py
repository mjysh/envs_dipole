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
        return action, value

def test():
    ############## Hyperparameters ##############
    policy_path = 'tools/bestpolicy_geo'
    Nobs = 5
    N1 = 128
    N2 = 128
    Naction = 1
    # env = fish.DipoleSingleEnv(paramSource = 'envParam_default')
    # env = fish.DipoleSingleEnv(paramSource = 'envParam_ego2sensorLRGradCFD')
    env = fish.DipoleSingleEnv(paramSource = 'envParam_geo1sensorCFD')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    n_episodes = 1          # num of episodes to run
    max_timesteps = 1900    # max timesteps in one episode
    render = True           # render the environment
    save_gif = False        # png images are saved in gif folder
    
    
    trained_policy = VRACER_NN(path = policy_path, Nobs = Nobs, N1 = N1, N2 = N2, Naction = Naction)
    beta_history = np.zeros((max_timesteps+1,n_episodes))
    x_history = np.zeros((max_timesteps+1,n_episodes))
    y_history = np.zeros((max_timesteps+1,n_episodes))
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        env = wrappers.Monitor(env, './Movies/test',force = True)
        obs = env.reset(position = [-9.0,-2.0,0.0], target = [-12.0,2.0], init_time = None)
        obs_his = obs
        env.done = False
        for t in range(max_timesteps):
            action = trained_policy.obs_to_act(obs)
            obs, reward, env.done, _ = env.step(action)
            ep_reward += reward
            # print(obs)
            if render:
                obs_his = np.vstack((obs_his,obs))
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
            beta_history[t,ep-1] = env.pos[-1]
            x_history[t,ep-1] = env.pos[0]
            y_history[t,ep-1] = env.pos[0]
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
        print('Episode: {}\tReward: {}'.format(ep, (ep_reward)))
        ep_reward = 0
        env.close()
if __name__ == '__main__':
    test()
    
    
