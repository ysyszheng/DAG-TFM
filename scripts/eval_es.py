import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
import csv
from agents.es import Net


class Evaluator(object):
    def __init__(self, cfgs):
        super(Evaluator, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed + 666)
        self.device = torch.device('cpu')

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.strategies = Net(num_agents=1, num_actions=1).to(self.device)
        self.strategies.load_state_dict(torch.load(cfgs.path.model_path))

    def evaluating(self):
        state = self.env.reset()
        
        progress_bar = tqdm(range(1, self.cfgs.eval.steps+1))
        for _ in progress_bar:
            action = self.strategies(torch.FloatTensor(state).to(torch.float64)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()

            _, max_reward = self.env.find_all_optim_action(action)

            next_state, reward, _, _ = self.env.step(action)

            dev = (max_reward - reward) / state
            state = next_state

            print(dev)
            print(np.mean(dev))
            print(np.amax(dev))
