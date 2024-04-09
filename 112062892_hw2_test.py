import numpy as np
import torch
from os.path import join

import torch

import warnings
import time
from scipy.stats import pearsonr
warnings.filterwarnings("ignore", category=DeprecationWarning)

from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros

from tensordict import TensorDict
import matplotlib.pyplot as plt
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import importlib
from scipy.stats import pearsonr
#hw2_train = importlib.import_module("GPU_ver2.py")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

model_name="112061588_hw2_train"
hw2_train = importlib.import_module(model_name)
FrameDownsample = hw2_train.FrameDownsample
MaxAndSkipEnv = hw2_train.MaxAndSkipEnv
Agent = hw2_train.Agent
wrap_environment = hw2_train.wrap_environment

# ----------------------Initialize the agent------------------------------
agent = Agent()
online_network_path = '112061588_hw2_data.dat'

agent.policy_net.load_state_dict(torch.load(online_network_path, map_location=torch.device('cpu')))
# --------------------Initialize environment---------------------------------
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]
#env = wrap_environment(COMPLEX_MOVEMENT)
env=gym_super_mario_bros.make("SuperMarioBros-v0") #(1,240,256,3)
episodes = 50
ep_rewards = []

for e in range(episodes):
    state = env.reset()
    #print(state.shape)
    #print(state.shape)
    total_reward = 0.0
    r=[]
    # Play the game!
    step=0
    while True:
        step+=1
        #print(state.shape)
        #env.render()
       # Run agent on the state
        action = agent.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)
        agent.mode=0
        correlation, p_value = pearsonr(state.flatten(), next_state.flatten())
        if correlation>0.9 and (len(r)==0 or step==(r[-1]+1)):
            r.append(step)
        else:
            r=[]
        if (len(r)>10):
            agent.mode=1
        # if ((next_state == state)/np.size(state)):
        #     print("qq")
        
        # Update state
        state = next_state
        total_reward += reward
        if done:
            break
        if info["flag_get"]:
            print("flag get!")
            #online_network_path = './112061588_hw2_data/SuperMarioBros-1-2-v0.dat'
            #agent.policy_net.load_state_dict(torch.load(online_network_path, map_location=torch.device('cpu')))
            break
        time.sleep(0.001)
    ep_rewards.append(total_reward)

env.close()
plt.figure(figsize=(10, 6))
plt.plot(ep_rewards, label='Episode Rewards', color='red')
print("50 rounds reward:",sum(ep_rewards)/episodes)

