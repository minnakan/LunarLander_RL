from ddpg import DDPG
from buffer import ReplayBuffer
import os

import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import datetime
import copy

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device('cuda:0')
    
    #Force CPU
    #device = torch.device('cpu')
else:
    device = torch.device('cpu')
print('Found device at: {}'.format(device))

env = gym.make('LunarLanderContinuous-v2',render_mode="human")
env.metadata['render_fps']=2000
config = {
    'dim_obs': 8,
    'dim_action': 2,
    'dims_hidden_neurons': (400, 200),
    'lr_actor': 0.001,
    'lr_critic': 0.005,
    'smooth': 0.01,
    'discount': 0.99,
    'sig': 0.01,
    'batch_size': 32,
    'replay_buffer_size': 20000,
    'seed': 1,
    'max_episode': 2000,
    'device':device
}

ddpg = DDPG(config).to(device)
buffer = ReplayBuffer(config)
train_writer = SummaryWriter(log_dir='tensorboard/ddpg_{date:%Y-%m-%d_%H_%M_%S}'.format(
                             date=datetime.datetime.now()))

steps = 0
for i_episode in range(config['max_episode']):
    obs = env.reset()[0]
    done = False
    truncated = False
    t = 0
    ret = 0.
    while done is False and truncated is False:
        env.render()

        obs_tensor = torch.tensor(obs).type(Tensor).to(device)

        action = ddpg.act_probabilistic(obs_tensor[None, :]).detach().cpu().numpy()[0, :]

        next_obs, reward, done, truncated,_ = env.step(action)

        buffer.append_memory(obs=obs_tensor,
                             action=torch.from_numpy(action),
                             reward=torch.from_numpy(np.array([reward/10.0])),
                             next_obs=torch.from_numpy(next_obs).type(Tensor),
                             done=done)

        ddpg.update(buffer)

        t += 1
        steps += 1
        ret += reward

        obs = copy.deepcopy(next_obs)

        if done or truncated:
            print("Episode {} return {}".format(i_episode, ret))
        train_writer.add_scalar('Performance/episodic_return', ret, i_episode)

env.close()
train_writer.close()


