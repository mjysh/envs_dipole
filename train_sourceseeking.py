# ------------------------------------------------------------------
# required package: openAI gym, PyTorch
# ------------------------------------------------------------------
import torch
import numpy as np
import os
from PPO_sourceseeking import PPO, Memory
import sourcefind_env as dipole

        
def main(k):
    path = './singledipole/200/{}'.format(k)
    if not os.path.exists(path):
        os.makedirs(path)
    ############## Hyperparameters ##############
    env_name = "singleDipole-v0" 
    dt = 0.1
    render = False              # render the environment in training if true
    solved_reward = 250         # stop training if avg_reward > solved_reward
    log_interval = 30           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 800         # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.4            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    fishflex = 0.1                  # amount of change allowed in vortex strength
    random_seed = None
    list_setting = ["decision time step: {}".format(dt), \
                    "fish flexibility: {}".format(fishflex), \
                    "max episodes: {}".format(max_episodes), \
                    "max time steps: {}".format(max_timesteps), \
                    "update time steps: {}".format(update_timestep), \
                    "standard deviation in actions: {}".format(action_std), \
                    "number of update epochs: {}".format(K_epochs), \
                    "clip parameter: {}".format(eps_clip), \
                    "discount factor: {}".format(gamma), \
                    "Adam optimizer lr: {}".format(lr), \
                    "Adam optimizer betas: {}".format(betas)]
    with open(path + "/setting.txt","w") as file_setting:
        file_setting.write("\n".join(list_setting))
    #############################################
    
    # creating environment
    env = dipole.DipoleSingleEnv()

    # set the length of an episode
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=max_timesteps)

    # get observation and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

#    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    file_logging = open(path + "/log.txt","w")
    file_logging.close()
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()

        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            # print(state)
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(float(reward))
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        # print(running_reward)
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
#            torch.save(ppo.policy.state_dict(), './PPO_Target_solved_{}.pth'.format(env_name))
            break
    
        # save every 50 episodes
        if i_episode % 50 == 0:
            torch.save(ppo.policy.state_dict(), path+'/PPO_{}_Target{:06d}.pth'.format(env_name,i_episode))
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = ((running_reward/log_interval))
            report = 'Episode {} \t Avg length: {} \t Avg reward: {}\n'.format(i_episode, avg_length, running_reward)
            print(report)
            with open(path + "/log.txt","a") as file_logging:
                file_logging.write(report)
            running_reward = 0
            avg_length = 0
    env.close()
        # ------------------------------------------------------------------
            
if __name__ == '__main__':
    print('single dipole 200')
    # training for 25 times
    for k in range(1,11):        
        main(k)

# wrap angles about a given center
def angle_normalize(x,center = 0):
    return (((x+np.pi-center) % (2*np.pi)) - np.pi+center)
