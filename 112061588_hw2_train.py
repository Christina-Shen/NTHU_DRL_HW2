# references and pretrained model from :https://github.com/roclark/super-mario-bros-dqn/blob/master/core/argparser.py

from os.path import join
from shutil import copyfile, move
import torch
from torch import save
from torch.optim import Adam
import random as rand

import cv2
import numpy as np
from collections import deque
from gym import make, ObservationWrapper, wrappers, Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import numpy as np
from random import random, randrange

import torch
import torch.nn as nn
from random import sample
from collections import deque
import math
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import os

import datetime
import gym_super_mario_bros

#----------------------enviroment wrapper-----------------------
class FrameDownsample(ObservationWrapper):
    def __init__(self, env):
        super(FrameDownsample, self).__init__(env)
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(84, 84, 1),
                                     dtype=np.uint8)
        self._width = 84
        self._height = 84

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,
                           (self._width, self._height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class MaxAndSkipEnv(Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class FireResetEnv(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        if len(env.unwrapped.get_action_meanings()) < 3:
            raise ValueError('Expected an action space of at least 3!')

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FrameBuffer(ObservationWrapper):
    def __init__(self, env, num_steps, dtype=np.float32):
        super(FrameBuffer, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = Box(obs_space.low.repeat(num_steps, axis=0),
                                     obs_space.high.repeat(num_steps, axis=0),
                                     dtype=self._dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low,
                                    dtype=self._dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(obs_shape[::-1]),
                                     dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class NormalizeFloats(ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 1350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info
        

def wrap_environment(action_space):
    env=gym_super_mario_bros.make("SuperMarioBros-v0")
    #env = make(environment)
    env = JoypadSpace(env, action_space)
    env = MaxAndSkipEnv(env)
    env = FrameDownsample(env)
    env = ImageToPyTorch(env)
    env = FrameBuffer(env, 4)
    env = NormalizeFloats(env)
    env = CustomReward(env)
    return env
# ------------------Memory buffer with priortized--------------
class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self._alpha = alpha
        self._capacity = capacity
        self._buffer = []
        self._position = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self._priorities.max() if self._buffer else 1.0

        batch = (state, action, reward, next_state, done)
        if len(self._buffer) < self._capacity:
            self._buffer.append(batch)
        else:
            self._buffer[self._position] = batch

        self._priorities[self._position] = max_prio
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, beta=0.4):
        if len(self._buffer) == self._capacity:
            prios = self._priorities
        else:
            prios = self._priorities[:self._position]

        probs = prios ** self._alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self._buffer), batch_size, p=probs)
        samples = [self._buffer[idx] for idx in indices]

        total = len(self._buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self._priorities[idx] = prio

    def __len__(self):
        return len(self._buffer)
    

#----------------------Agent policy /target network-------------
class Agentnet(nn.Module):
    def __init__(self):
        super(Agentnet, self).__init__()
        self._input_shape =(4, 84, 84)
        self._num_actions = 12
        
        self.features = nn.Sequential(
            nn.Conv2d(self._input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 12)
        )
        #print(self.feature_size)

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)

    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)
#----------------------Agent choose action----------------------
class Agent:
    def __init__(self):
        
        self.state_dim = (4, 84, 84)
        self.action_dim = 12
        self.test_mode = 1
        self.device=torch.device('cpu')
        self.mode=0

    def choose_action(self, state):
        if self.mode==1:
             action = rand.choice([2,3,4])
             #print("choose action 2")
             return action
        if random() > self.cur_eplison or self.test_mode == 1:
            state = torch.FloatTensor(np.float32(state)) \
                .unsqueeze(0).to(self.device)
            q_value = self.policy_net.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = randrange(self.action_dim)
        # if self.test_mode == 1:
        #     state_v = torch.tensor(np.array([state], copy=False))
        #     q_value = self.policy_net.forward(state_v).data.numpy()
        #     q_value = q_value[0]
        return action

#---------------------Agent cache/ recall memory---------------
class Agent(Agent):  # subclassing for continuity
    def __init__(self):
        super().__init__()
        self.policy_net = Agentnet().to(self.device)
        self.critic_net= Agentnet().to(self.device)

        self.optimizer=Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = PrioritizedBuffer(100000)
    def cache_memory(self,state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
#---------------------Agent update parameter----------------------
class Agent(Agent):  # subclassing for continuity
    def __init__(self):
        super().__init__()
        #self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.cur_eplison=1
        self.eplison_decay=100000
        self.eplison_min=0.01
        self.eplison_start=1
        self.beta_start=0.4
        self.curr_step=0
        self.beta_frame=10000
    def update_parameter(self,e):
        self.cur_eplison = self.eplison_min + (self.eplison_start - self.eplison_min) * math.exp(-1 * ((e + 1) / self.eplison_decay))
        if len(self.replay_buffer) > self.batch_size:
            self.beta = self.beta_start+e*(1-self.beta_start)/self.beta_frame
        else:
            self.beta = self.beta_start
        self.beta=min(1,self.beta)
        self.curr_step+=1
        if len(self.replay_buffer) > 10000:
            if not self.curr_step % self.target_update_freq:
                self.critic_net.load_state_dict(self.policy_net.state_dict())

# --------------------------Agent learn and compute td_loss-------------------------------

class Agent(Agent):  # subclassing for continuity
    def __init__(self):
        super().__init__()
        self.gamma=0.99
        self.batch_size=32
        #self.optimizer=Adam()
        #self.optimizer=Adam(self.policy_net.parameters(), lr=1e-4)
        self.target_update_freq=1000
    def compute_q_values(self,state,action):
        q_values=self.policy_net.forward(state)
        q_values = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return q_values
    def compute_max_target(self,next_state):
        next_q_values = self.critic_net.forward(next_state)
        next_q_value = next_q_values.max(1)[0]
        return next_q_value
    def compute_td_loss(self):
        batch = self.replay_buffer.sample(self.batch_size, self.beta)
        state, action, reward, next_state, done, indices, weights = batch
        state = Variable(FloatTensor(np.float32(state))).to(self.device)
        next_state = Variable(FloatTensor(np.float32(next_state))).to(self.device)
        action = Variable(LongTensor(action)).to(self.device)
        reward = Variable(FloatTensor(reward)).to(self.device)
        done = Variable(FloatTensor(done)).to(self.device)
        weights = Variable(FloatTensor(weights)).to(self.device)

        q_value= self.compute_q_values(state,action)
        next_q_value = self.compute_max_target(next_state)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()
        return prios,loss,indices
    def learn(self):
        self.optimizer.zero_grad()
        prios,loss,indices=self.compute_td_loss()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()
#----------------------start train------------------------------------
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
if __name__ == '__main__':


    env = wrap_environment(COMPLEX_MOVEMENT)
    agent = Agent()
    online_network_path = './112061588_hw2_data/SuperMarioBros-1-1-v0.dat'
    agent.policy_net.load_state_dict(torch.load(online_network_path, map_location=torch.device('cpu')))
    agent.critic_net.load_state_dict(torch.load(online_network_path, map_location=torch.device('cpu')))
    agent.test_mode = 0
    #------------load pretrained parameter---------

    #info = TrainInformation()
    flag=1
    dir_name = "112061588_hw2_data"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for episode in range(NUM_EPISODES):
        episode_reward = 0.0
        state = env.reset()

        while True:
            agent.update_parameter(episode)
            #----------------start learn----------------------------------
            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)

            agent.cache_memory(state, action, reward, next_state, done)

            agent.learn()


            #-----------------update state reward------------------------
            state = next_state
            episode_reward += reward
            #-----------------if done---------------------------
            if info['flag_get']:
                flag+=1
                break
            if done : 
                break
            #------------------print result & save------------------------
        if episode%5==0 or (episode == NUM_EPISODES - 1):
            if flag==1:
                online_file_name = "./112061588_hw2_data/online_network_1.pt"
                torch.save(agent.policy_net.state_dict(), online_file_name)
            if flag==2:
                online_file_name = "./112061588_hw2_data/online_network_2.pt"
                torch.save(agent.policy_net.state_dict(), online_file_name)
            # target_file_name = f"./112061588_hw2_data/target_network.pt"
            # torch.save(agent.net.target.state_dict(), target_file_name)
            print(
                    f"Episode {episode} - "
                    f"Step {agent.curr_step} - "
                    f"Mean Reward {episode_reward} - "
                    f"Time {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

    env.close()