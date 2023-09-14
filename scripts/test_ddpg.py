import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from agents.ddpg import DDPG
from utils.utils import fix_seed
from tqdm import tqdm
from utils.utils import log
import csv

class Tester(object):
    def __init__(self, cfgs):
        super(Tester, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs['seed'] + 666)

        self.env = gym.make('gym_dag_env-v0',
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, b=cfgs.b, is_burn=cfgs.is_burn
        )
        self.agent = DDPG(1, 1, cfgs.lr, cfgs.gamma, cfgs.tau,
            cfgs.batch_size, cfgs.epsilon_min, cfgs.epsilon_decay
        )
        self.rewards_lst = []
        self.agent.actor.load_state_dict(torch.load('./models/actor_150.pth')) # !!!!!!!!!
    
    def testing(self):
      state = self.env.reset()
      state = torch.tensor(state, dtype=torch.float32)

      with open(f'{self.cfgs["fn"]}_{self.cfgs["mode"]}.csv', 'w', newline='') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow(["step", "private value", "transaction fee", "incentive awareness", "revenue", "probability", "miner number"])

      progress_bar = tqdm(range(1, self.cfgs['test_steps']+1))
      for step in progress_bar:
          self.cfgs['num_miners'] = self.env.num_miners
          action = np.zeros_like(state)

          for i in range(len(state)):
            if i == self.cfgs['optim_idx'] and self.cfgs['mode'] == 'verify':
              action[i] = self.agent.select_action_all_with_exp(state[i], self.cfgs['verify_std'])
            else:
              action[i] = self.agent.select_action(state[i])
        
          if self.cfgs['mode'] == 'optim':
            optim_action = self.env.find_optim_action(action, self.cfgs['optim_idx'], self.cfgs['optim_cnt'])
            action[self.cfgs['optim_idx']] = optim_action

          next_state, reward, done, info = self.env.step(action)

          with open(f'{self.cfgs["fn"]}_{self.cfgs["mode"]}.csv', 'a', newline='') as csvfile:
            for s, a, r, p in zip(state.data.numpy().flatten(), action, reward, info['probabilities']):
              writer = csv.writer(csvfile)
              writer.writerow([step, s, a, a/s, r, p, self.cfgs['num_miners']])

          next_state = torch.tensor(next_state, dtype=torch.float32)
          state = next_state

          if done:
              state = self.env.reset()
              state = torch.tensor(state, dtype=torch.float32)

          self.rewards_lst.append(sum(reward))
          progress_bar.set_description(f"total revenue: {self.rewards_lst[-1]}")
      log(f"total revenue: {self.rewards_lst}")
 