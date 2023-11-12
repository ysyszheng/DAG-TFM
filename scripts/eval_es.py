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
        avg_regret_list_random = []
        avg_regret_list_es = []

        for _ in range(self.cfgs.eval.test_round):
            state = self.env.reset()
            action_random = np.random.random(state.shape) * state
            action_es = self.strategies(torch.FloatTensor(state).to(torch.float64)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()

            _, opt_reward_random = self.env.find_all_optim_action(action_random)
            reward_random, _ = self.env.calculate_rewards(action_random)
            _, opt_reward_es = self.env.find_all_optim_action(action_es)
            reward_es, _ = self.env.calculate_rewards(action_es)
            dev_random = (opt_reward_random - reward_random) / state
            dev_es = (opt_reward_es - reward_es) / state
            nash_apr_random = np.max(dev_random)
            nash_apr_es = np.max(dev_es)
            avg_regret_random = np.mean(dev_random)
            avg_regret_es = np.mean(dev_es)
            avg_regret_list_random.append(avg_regret_random)
            avg_regret_list_es.append(avg_regret_es)
            print(f'Random startegies:  nash apr: {nash_apr_random},\tavg regert: {avg_regret_random}')
            print(f'NES startegies:     nash apr: {nash_apr_es},\tavg regert: {avg_regret_es}')

