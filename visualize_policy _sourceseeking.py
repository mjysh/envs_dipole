import time
from PPO_sourceseeking import PPO, Memory
from PIL import Image
import torch
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
device = torch.device("cpu")
def visualize_policy():
    ############## Hyperparameters ##############
    env_name = "singleDipole-v0"
    # creating environment
#    env = gym.make(env_name)
    state_dim = 1
    action_dim = 1
    action_std = 0.01
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################
#    n_episodes = 10

    filename = 'PPO_{}_Target{:06d}.pth'.format(env_name,400)

    directory = "./singledipole/200/1/"
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory+filename,map_location=device))
    
    Nres = 101
    test_betas = torch.linspace(-1,1,Nres).to(device)

    
    fig = plt.figure()
    ax = plt.axes()

    action = np.zeros((Nres,))
    for i in range(Nres):
        test_input = test_betas[i].reshape(1,-1).to(device)
        # action[i] = ppo.select_action(test_input, memory)
        action[i] = ppo.policy_old.actor(test_input).squeeze().detach().cpu().numpy()
    from fractions import Fraction
    ax.plot(test_betas.data.cpu().numpy(),action)
    # ax.set_aspect('equal')
    plt.show()
    return 0

if __name__ == '__main__':
    visualize_policy()
