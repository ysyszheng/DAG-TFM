import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from agents.ddpg import DDPG
from utils.replay_buffer import ReplayBuffer
from utils.fix_seed import fix_seed
from tqdm import tqdm
from utils.log import log
import csv

class Trainer(object):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.cfg = cfg
        fix_seed(cfg['seed'])

        self.env = gym.make(
            'gym_dag_env-v0',
            max_agents_num=cfg['max_agents_num'],
            lambd=cfg['lambd'],
            delta=cfg['delta'],
            b=cfg['b'],
            is_burn=cfg['is_burn']
        )
        self.agent = DDPG(
            1,
            1,
            cfg['lr'],
            cfg['gamma'],
            cfg['tau'],
            cfg['batch_size'],
            cfg['epsilon_min'],
            cfg['epsilon_decay']
        )
        self.replay_buffer = ReplayBuffer()
        self.rewards_lst = []

    def training(self):
      state = self.env.reset()
      state = torch.tensor(state, dtype=torch.float32)

      with open(f'{self.cfg["fn"]}_{self.cfg["mode"]}.csv', 'w', newline='') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow(["step", "private value", "transaction fee", "incentive awareness", "revenue", "probability", "miner number"])

      progress_bar = tqdm(range(1, self.cfg['steps']+1))
      for step in progress_bar:
          self.cfg['num_miners'] = self.env.num_miners
          action = np.zeros_like(state)
          for i in range(len(state)):
              action[i] = self.agent.select_action_with_exp(state[i], self.cfg['exp_std'])
          next_state, reward, done, info = self.env.step(action)

          with open(f'{self.cfg["fn"]}_{self.cfg["mode"]}.csv', 'a', newline='') as csvfile:
            for s, a, r, p in zip(state.data.numpy().flatten(), action, reward, info['probabilities']):
              writer = csv.writer(csvfile)
              writer.writerow([step, s, a, a/s, r, p, self.cfg['num_miners']])

          next_state = torch.tensor(next_state, dtype=torch.float32)
          for i in range(len(state)):
              self.replay_buffer.add((state[i], action[i], reward[i]))
          state = next_state

          if len(self.replay_buffer) > self.cfg['batch_size']:
              self.agent.update(self.replay_buffer, self.cfg['iterations'])
          if done:
              state = self.env.reset()
              state = torch.tensor(state, dtype=torch.float32)

          self.rewards_lst.append(sum(reward))
          progress_bar.set_description(f"total revenue: {self.rewards_lst[-1]}")

          if step % self.cfg['save_freq'] == 0:
              torch.save(self.agent.actor.state_dict(), f'./models/actor_{step}.pth')
              torch.save(self.agent.critic.state_dict(), f'./models/critic_{step}.pth')
              np.save(f'./rewards/rewards_{step}.npy', self.rewards_lst)
