import numpy as np
from envs.DAGEnv import DAGEnv
from agents.dnn import DoubleNN, ProbNet
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
    self.dnn = DoubleNN(num_agents=cfgs.max_agents_num, num_actions=cfgs.max_agents_num, lr_1=cfgs.train.lr_1, lr_2=cfgs.train.lr_2).to(self.device)

  def training(self):      
    state = self.env.reset()
    prob_loss_list = []
    reward_loss_list = []

    progress_bar = tqdm(range(1, self.cfgs.train.epochs+1))
    for epoch in progress_bar:
      batch_probs = torch.zeros((self.cfgs.train.batch_size,self.cfgs.max_agents_num)).to(self.device)
      true_probs = np.zeros((self.cfgs.train.batch_size,self.cfgs.max_agents_num))
      batch_rewards = torch.zeros((self.cfgs.train.batch_size,self.cfgs.max_agents_num)).to(self.device)
      optimal_rewards = np.zeros((self.cfgs.train.batch_size,self.cfgs.max_agents_num))

      for i in range(self.cfgs.train.batch_size):
        batch_probs[i,:], actions, batch_rewards[i,:] = self.dnn(torch.FloatTensor(state).to(self.device), self.env.num_miners)

        for idx in range(self.cfgs.max_agents_num):
          _, optimal_reward = self.env.find_optim_action(actions=actions.detach().cpu().numpy(), idx=idx, cnt=self.cfgs.train.cnt)
          optimal_rewards[i,idx] = optimal_reward

        next_state, rewards, _, info = self.env.step(actions.detach().cpu().numpy())
        true_probs[i,:] = info["probabilities"]
        state = next_state

        optimal_rewards[i,:] = np.where(optimal_rewards[i,:] >= rewards, optimal_rewards[i,:], rewards)

      self.dnn.learn(batch_probs, torch.FloatTensor(true_probs).to(self.device), batch_rewards, torch.FloatTensor(optimal_rewards).to(self.device))
      
      prob_loss_list.append(self.dnn.prob_loss.item())
      reward_loss_list.append(self.dnn.rev_loss.item())

      if epoch % self.cfgs.train.save_freq == 0:
        np.save(self.cfgs.path.prob_loss_path, np.array(prob_loss_list))
        np.save(self.cfgs.path.reward_loss_path, np.array(reward_loss_list))
        torch.save(self.dnn.state_dict(), self.cfgs.path.model_path)

      progress_bar.set_description(f'Episode {epoch}, prob loss={prob_loss_list[-1]}, rewards loss={reward_loss_list[-1]}')
