# from PPO_dipole_observation import PPO, Memory
from PIL import Image
import torch
import numpy as np
import sourcefind_env as swimmer
# import dipole_observation as fish
import time
from gym import wrappers
device = torch.device("cpu")

def test():
    ############## Hyperparameters ##############
    targetpos = np.array([0.,3.])
    env = swimmer.DipoleSingleEnv(paramSource = 'envParam_sourceseeking') 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    n_episodes = 1          # num of episodes to run
    max_timesteps = 2000    # max timesteps in one episode
    render = True           # render the environment
    
    beta_history = np.zeros((max_timesteps+1,n_episodes))
    x_history = np.zeros((max_timesteps+1,n_episodes))
    y_history = np.zeros((max_timesteps+1,n_episodes))
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        env = wrappers.Monitor(env, './Movies/sourceseekingtest',force = True)
        state = env.reset(position = [-0.,0.5,np.pi/2], target = [8.,0.])
        state_his = state
        env.done = False
        for t in range(max_timesteps):
            # action = ppo.select_action(state, memory)
            print(state)
            action = -np.sign(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            # print(state)
            if render:
                state_his = np.vstack((state_his,state))
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
#        ax1.plot(T[0],state_his[0,0],'ro',T,state_his[:,0],'k')
#        plt.show()
#        ax1.set_ylabel('orientation',size = 14)
#        ax1.set_title('Trained Policy',size = 20)
#        
#        
#        ax2.plot(T[0],state_his[0,2],'ro',T,state_his[:,2],'k')
#        plt.show()
#        ax2.set_ylabel('head angle',size = 14)
#        
#        ax3.plot(T[0],state_his[0,1],'ro',T,state_his[:,1],'k')
#        plt.show()
#        ax3.set_ylabel('tail angle',size = 14)
#        ax3.set_xlabel('time',fontsize = 14)
#        
#        fig,ax = plt.subplots()
#        
#        ax.plot(state_his[0,2],state_his[0,1],'ro',state_his[:,2],state_his[:,1],'k')
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
    
    