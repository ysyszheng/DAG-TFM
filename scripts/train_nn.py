import numpy as np
from envs.DAGEnv import DAGEnv
from models.nn import Net
from utils.utils import log, fix_seed
from tqdm import tqdm
from easydict import EasyDict as edict
import csv
import torch
import gym

class Trainer(object):
  def __init__(self, cfgs: edict):
    super(Trainer, self).__init__()
    self.cfgs = cfgs
    fix_seed(cfgs.seed)

    self.env = gym.make('gym_dag_env-v0', 
        fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
        lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
    )
    self.nn = Net(num_agents=cfgs.max_agents_num, num_actions=cfgs.max_agents_num, lr=cfgs.train.lr)

  def training(self):
    state = self.env.reset()
    sw_list = []
    loss_list = []

    progress_bar = tqdm(range(1, self.cfgs.train.epochs+1))
    for epoch in progress_bar:
      actions = self.nn(torch.FloatTensor(state))
      optimal_rewards = np.zeros(self.cfgs.max_agents_num)
      optimal_actions = np.zeros(self.cfgs.max_agents_num)

      for idx in range(self.cfgs.max_agents_num):
        optimal_action, optimal_reward = self.env.find_optim_action(actions=actions.detach().numpy(), 
                idx=idx, cnt=self.cfgs.train.cnt)
        optimal_actions[idx] = optimal_action
        optimal_rewards[idx] = optimal_reward

      next_state, rewards, _, _ = self.env.step(actions.detach().numpy())
      optimal_actions = np.where(optimal_rewards >= rewards, optimal_actions, actions.detach().numpy())
      optimal_rewards = np.where(optimal_rewards >= rewards, optimal_rewards, rewards)

      # self.nn.learn(rewards, optimal_rewards)
      self.nn.learn(actions, torch.FloatTensor(optimal_actions))
      
      state = next_state
      sw_list.append(rewards.sum())
      loss_list.append(self.nn.loss.item())

      if epoch % self.cfgs.train.save_freq == 0:
        np.save(self.cfgs.path.sw_path, np.array(sw_list))
        np.save(self.cfgs.path.loss_path, np.array(loss_list))
        torch.save(self.nn.state_dict(), self.cfgs.path.model_path)

      progress_bar.set_description(f'Episode {epoch}, loss={self.nn.loss.item()}')
