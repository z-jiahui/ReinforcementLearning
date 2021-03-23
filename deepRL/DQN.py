import numpy as np 
from matplotlib import pyplot as plt 
import gym
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from itertools import count
import time

#Replay Memory
class ReplayMemory:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []

  def push(self, transition):
    self.memory.append(transition)
    if len(self.memory) > self.capacity :
      del self.memory[0]
  
  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)
  
  def __len__(self):
    return len(self.memory)

class DQN(nn.Module):
  def __init__(self, obs_dims, act_dims):
    super(DQN,self).__init__()
    self.obs_dims = obs_dims
    self.act_dims = act_dims

    # self.conv1 = nn.Conv2d(self.input_shape[0], out_channels=32, kernel_size=8, stride=4)
    # self.conv2 = nn.Conv2d(32, 64, 4, 2)
    # self.conv3 = nn.Conv2d(64, 64, 3, 1)

    # self.fc1 = nn.Linear(self.feature_size(), 512)
    # self.fc2 = nn.Linear(512, self.num_actions)
    self.model = nn.Sequential(
                      nn.Linear(self.obs_dims, 50),
                      nn.ReLU(),
                      nn.Linear(50, 50),
                      nn.ReLU(),
                      nn.Linear(50, self.act_dims))

  def forward(self, x):
    return self.model(x)

  # def feature_size(self):
  #   return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(-1,1).size(0)
  
#超参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
GAMMA = 0.99
LR = 1e-4
TARGET_NET_UPDATE_FREQ = 20
MEMORY_SIZE = 1E6
BATCH_SIZE = 64
LEARN_START = 1E4
MAX_FRAMES = 1E6
LEARNING_TARE = 0.005
MAX_EPISODE = 1000

class Model():
  def __init__(self, env):
    self.device = DEVICE
    self.eps = 0.1
    self.gamma = GAMMA
    self.memory_size = MEMORY_SIZE
    self.batch_size = BATCH_SIZE
    self.target_net_update_freq = TARGET_NET_UPDATE_FREQ
    self.env = env
    self.obs_dim = env.observation_space.shape[0]
    self.act_dim = env.action_space.n
    self.learning_rate = LEARNING_TARE
    self.max_episode = MAX_EPISODE
    self.target_update_freq = TARGET_NET_UPDATE_FREQ

  def init_parameters(self):
    self.memory = ReplayMemory(self.memory_size)

    self.eval_net = DQN(self.obs_dim, self.act_dim)#.to(self.device)
    self.eval_net.eval()
    self.eval_net.share_memory()

    self.target_net = DQN(self.obs_dim, self.act_dim)#.to(self.device)
    self.target_net.load_state_dict(self.eval_net.state_dict())
    self.target_net.eval()

    self.optimizer = torch.optim.Adam(params=self.eval_net.parameters(),lr=self.learning_rate)
    self.loss = nn.MSELoss()

  def process_state(self, state):
    return torch.from_numpy(state).to(torch.float32)
  
  def compute_action(self, process_state):
    #无梯度跟踪时计算
    with torch.no_grad():
      value = self.eval_net(process_state).detach().numpy()
    return value

  def select_action(self, process_state):
    values = self.compute_action(process_state)
    if np.random.random() < self.eps:
      return np.random.choice(self.act_dim)
    else:
      return np.argmax(values)

  def train(self):
    update_count = 0
    for i in range(self.max_episode):
      state = self.env.reset()
      processed_state = self.process_state(state)
      for t in count():
        action = self.select_action(processed_state)
        next_state, reward, done, info = self.env.step(action)
        processed_next_state = self.process_state(next_state)
        #processed_state = process_next_state
        self.memory.push((processed_state.unsqueeze(0), [action], reward, processed_next_state.unsqueeze(0), done))
        processed_state = processed_next_state
        if done:  #本轮已结束
          print("this episode done")
          break
        #样本未达到一定值，不训练
        if len(self.memory) < 2000:
          continue
        elif len(self.memory) == 2000:
          print("Start training")
        #处理sample batch
        sample_batch = self.memory.sample(self.batch_size)
        prc_state_batch, action_batch, reward_batch, prc_next_state_batch, done_batch = \
          list(zip(*sample_batch))
        prc_state_batch = torch.cat(prc_state_batch,0)
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.tensor(reward_batch)
        prc_next_state_batch = torch.cat(prc_next_state_batch)
        done_batch = torch.tensor(done_batch)
        #计算目标Q值
        
        with torch.no_grad():
          output = self.eval_net(prc_state_batch)
          maxvalue = torch.max(output, 1)[1]
        Q_next = torch.max(self.target_net(prc_next_state_batch).detach(), 1)[0];
        Q_target = reward_batch + (self.gamma * Q_next) * (~done_batch)
        Q_target = Q_target.unsqueeze(1)
        #计算估计Q值
        self.eval_net.train()
        #Q_eval = self.eval_net(prc_state_batch)[:,action_batch]
        Q_eval = self.eval_net(prc_state_batch).gather(1,action_batch)
        #梯度优化
        loss = self.loss(Q_eval, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), 5)
        self.optimizer.step()
      #更新targetNet的参数
      if i % self.target_update_freq == 0:
        self.target_net.load_state_dict(self.eval_net.state_dict())
        print("update target")

  def evaluate(self, episode = 50):
    reward = []
    for _ in range(episode):
      state = self.env.reset()
      prc_state = self.process_state(state)
      action = self.select_action(prc_state)
      ep_reward = 0
      done = False
      while not done:
        self.env.render()
        #time.sleep(0.1)
        state, r, done, info = self.env.step(action)
        prc_state = self.process_state(state)
        action = self.select_action(prc_state)
        ep_reward += r
      reward.append(ep_reward)
    return np.mean(reward)

if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  Trainer = Model(env)
  Trainer.init_parameters()
  Trainer.train()
  Trainer.evaluate()
  Trainer.env.close()
