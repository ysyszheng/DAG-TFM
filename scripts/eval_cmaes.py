import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
import csv
from agents.cmaes import Net
import multiprocessing as mp
from copy import deepcopy
from typing import List


class Evaluator(object):
    def __init__(self, cfgs):
        super(Evaluator, self).__init__()
        self.cfgs = cfgs
        self.device = torch.device('cpu')
        fix_seed(self.cfgs.seed + 666)

        self.env: DAGEnv = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.strategies = Net(num_agents=1, num_actions=1).to(self.device)
        self.strategies.load_state_dict(torch.load(cfgs.path.model_path))

    
    def worker(self, i: int):
        local_env = deepcopy(self.env)
        torch.manual_seed(i)
        np.random.seed(i)

        state = local_env.reset()
        state0 = state[0]
        regret_random, regret_es = 0, 0

        for _ in range(self.cfgs.eval.expect_time):
            state[0], local_env.state[0] = state0, state0
            action_random = np.random.random(state.shape) * state
            action_es = self.strategies(torch.FloatTensor(state)\
                .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            
            _, opt_reward_random = local_env.find_optim_action(action_random)
            reward_random, _ = local_env.calculate_rewards(action_random)
            _, opt_reward_es = local_env.find_optim_action(action_es)
            reward_es, _ = local_env.calculate_rewards(action_es)
            
            regret_random += (opt_reward_random - reward_random[0]) / state[0]
            regret_es += (opt_reward_es - reward_es[0]) / state[0]
            state = local_env.reset()

        regret_random /= self.cfgs.eval.expect_time
        regret_es /= self.cfgs.eval.expect_time
        return regret_random, regret_es


    def evaluating(self):
        regret_list_random = []
        regret_list_es = []

        for i in range(self.cfgs.eval.query_time):
            self.env.reset()
            with mp.Pool(processes=mp.cpu_count()) as p:
                results = p.starmap(self.worker, [(j+i*mp.cpu_count(),) for j in range(mp.cpu_count())])
                for result in results:
                    regret_list_random.append(result[0])
                    regret_list_es.append(result[1])

            # state = self.env.reset()
            # state0 = state[0]
            # regret_random, regret_es = 0, 0

            # for _ in range(self.cfgs.eval.expect_time):
            #     state[0], self.env.state[0] = state0, state0
            #     action_random = np.random.random(state.shape) * state
            #     action_es = self.strategies(torch.FloatTensor(state)\
            #     .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
                
            #     _, opt_reward_random = self.env.find_optim_action(action_random)
            #     reward_random, _ = self.env.calculate_rewards(action_random)
            #     _, opt_reward_es = self.env.find_optim_action(action_es)
            #     reward_es, _ = self.env.calculate_rewards(action_es)
                
            #     regret_random = (opt_reward_random - reward_random[0]) / state[0]
            #     regret_es = (opt_reward_es - reward_es[0]) / state[0]
            #     state = self.env.reset()

            # regret_random, regret_es = regret_random / self.cfgs.eval.expect_time, regret_es / self.cfgs.eval.expect_time
            # regret_list_random.append(regret_random)
            # regret_list_es.append(regret_es)

            print(f'len: {len(regret_list_random)}, Random: ea-epsilon: {sum(regret_list_random) / len(regret_list_random)}, interim epsilon: {max(regret_list_random)}')
            print(f'len: {len(regret_list_es)}, CMAES: ea-epsilon: {sum(regret_list_es) / len(regret_list_es)}, interim epsilon: {max(regret_list_es)}')
