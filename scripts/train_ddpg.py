import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from agents.ddpg import DDPG
from utils.replay_buffer import ReplayBuffer
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
from easydict import EasyDict as edict
import csv

class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = DDPG(1, 1, cfgs.model.actor_lr, cfgs.model.critic_lr, cfgs.model.gamma, cfgs.model.tau,
            cfgs.model.std, cfgs.model.batch_size, cfgs.model.epsilon_min, cfgs.model.epsilon_decay
        )
        self.replay_buffer = ReplayBuffer()

    def training(self):
      state = self.env.reset()
      state = normize(state, self.env.state_mean, self.env.state_std)
      rewards_list = []
      sw_list = []

      with open(self.cfgs.path.log_path, 'w', newline='') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow(["update step", "private value", "tx fee", "incentive awareness", 
                  "revenue", "probability", "miner numbers"])

      progress_bar = tqdm(range(1, self.cfgs.train.steps+1))
      for step in progress_bar:
          action = np.zeros_like(state)

          for i in range(len(state)):
              action[i] = self.agent.select_action_with_exploration(state[i])

          state = unormize(state, self.env.state_mean, self.env.state_std)
          action = action * state
          
          next_state, reward, _, info = self.env.step(action)

          with open(self.cfgs.path.log_path, 'a', newline='') as csvfile:
            for s, a, r, p in zip(state, action, reward, info['probabilities']):
              writer = csv.writer(csvfile)
              writer.writerow([step, s, a, a/s, r, p, self.env.num_miners])

          for i in range(len(state)):
              self.replay_buffer.add((state[i], action[i], reward[i]))
          
          state = normize(next_state, self.env.state_mean, self.env.state_std)

          if len(self.replay_buffer) > self.cfgs.model.batch_size:
              self.agent.update(self.replay_buffer, self.cfgs.model.iterations)

          rewards_list.extend(reward)
          sw_list.append(sum(reward))

          if step % self.cfgs.train.save_freq == 0:
              np.save(self.cfgs.path.rewards_path, np.array(rewards_list))
              np.save(self.cfgs.path.sw_path, np.array(sw_list))
              torch.save(self.agent.actor.state_dict(), self.cfgs.path.actor_model_path)
              torch.save(self.agent.critic.state_dict(), self.cfgs.path.critic_model_path)

          progress_bar.set_description(f"step: {step}, sw: {sum(reward)}")