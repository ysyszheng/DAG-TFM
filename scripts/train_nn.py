import numpy as np
from envs.DAGEnv import DAGEnv
from agents.nn import Net
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

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.env = gym.make('gym_dag_env-v0', 
        fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
        lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
    )
    self.nn = Net(num_agents=cfgs.max_agents_num, num_actions=cfgs.max_agents_num, lr=cfgs.train.lr).to(self.device)

  def training(self):
    state = self.env.reset()
    sw_list = []
    loss_list = []
    delta_rewards_list = []

    progress_bar = tqdm(range(1, self.cfgs.train.epochs+1))
    for epoch in progress_bar:
      batch_actions = torch.zeros((self.cfgs.train.batch_size,self.cfgs.max_agents_num)).to(self.device)
      batch_rewards = np.zeros((self.cfgs.train.batch_size,self.cfgs.max_agents_num))
      optimal_rewards = np.zeros((self.cfgs.train.batch_size,self.cfgs.max_agents_num))
      optimal_actions = np.zeros((self.cfgs.train.batch_size,self.cfgs.max_agents_num))

      for i in range(self.cfgs.train.batch_size):
        batch_actions[i,:] = self.nn(torch.FloatTensor(state).to(self.device))
        actions = batch_actions[i,:].detach().cpu().numpy()

        for idx in range(self.cfgs.max_agents_num):
          optimal_action, optimal_reward = self.env.find_optim_action(
              actions=actions, idx=idx, cnt=self.cfgs.train.cnt)
          optimal_actions[i,idx] = optimal_action
          optimal_rewards[i,idx] = optimal_reward

        next_state, rewards, _, _ = self.env.step(actions)
        batch_rewards[i,:] = rewards
        state = next_state

        optimal_actions[i,:] = np.where(optimal_rewards[i,:] >= rewards, optimal_actions[i,:], actions)
        optimal_rewards[i,:] = np.where(optimal_rewards[i,:] >= rewards, optimal_rewards[i,:], rewards)

      self.nn.learn(batch_actions, torch.FloatTensor(optimal_actions).to(self.device))
      
      sw_list.append(rewards.sum())
      loss_list.append(self.nn.loss.item())
      delta_rewards = (optimal_rewards - batch_rewards)
      delta_rewards_list.append(np.mean(np.max(delta_rewards, axis=1)))

      if epoch % self.cfgs.train.save_freq == 0:
        np.save(self.cfgs.path.sw_path, np.array(sw_list))
        np.save(self.cfgs.path.loss_path, np.array(loss_list))
        np.save(self.cfgs.path.delta_rewards_path, np.array(delta_rewards_list))
        torch.save(self.nn.state_dict(), self.cfgs.path.model_path)

      progress_bar.set_description(f'Episode {epoch}, loss={self.nn.loss.item()}, delta rewards={delta_rewards_list[-1]}')
