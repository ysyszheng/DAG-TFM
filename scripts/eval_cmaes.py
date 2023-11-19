import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
import csv
from agents.cmaes import Net


class Evaluator(object):
    def __init__(self, cfgs):
        super(Evaluator, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed + 666)
        self.device = torch.device('cpu')

        self.env: DAGEnv = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.strategies = Net(num_agents=1, num_actions=1).to(self.device)
        self.strategies.load_state_dict(torch.load(cfgs.path.model_path))

    def evaluating(self):
        mean_regret_list_random = []
        mean_regret_list_es = []

        for _ in range(self.cfgs.eval.test_round):
            state = self.env.reset()
            action_random = np.random.random(state.shape) * state
            action_es = self.strategies(torch.FloatTensor(state)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            
            _, opt_reward_random = self.env.find_optim_action(action_random)
            reward_random, _ = self.env.calculate_rewards(action_random)
            _, opt_reward_es = self.env.find_optim_action(action_es)
            reward_es, _ = self.env.calculate_rewards(action_es)
            
            dev_random = (opt_reward_random - reward_random[0]) / state[0]
            dev_es = (opt_reward_es - reward_es[0]) / state[0]
            mean_regret_list_random.append(dev_random)
            mean_regret_list_es.append(dev_es)

            print(f'Random: mean regert: {sum(mean_regret_list_random) / len(mean_regret_list_random)}, max regret: {max(mean_regret_list_random)}')
            print(f'CMAES: mean regert: {sum(mean_regret_list_es) / len(mean_regret_list_es)}, max regret: {max(mean_regret_list_es)}')
