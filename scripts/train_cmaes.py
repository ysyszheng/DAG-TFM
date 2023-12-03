import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cma import CMAEvolutionStrategy
import torch.nn as nn
from envs.DAGEnv import DAGEnv
from agents.cmaes import Net
from utils.utils import log, fix_seed
from tqdm import tqdm
from easydict import EasyDict as edict
import csv
import torch
import gym
import torch.multiprocessing as mp
from copy import deepcopy
import matplotlib.pyplot as plt
import ast
import time
from typing import List, Dict


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        self.cfgs.train.hidden_layer_size = ast.literal_eval(self.cfgs.train.hidden_layer_size)
        self.hls = self.cfgs.train.hidden_layer_size
        fix_seed(cfgs.seed)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.strategy = Net(num_agents=1, num_actions=1, hidden_layer_size=self.hls).\
          to(self.device)
        self.strategies = []

        self.env: DAGEnv = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )


    def reshape_params(self, params):
        if isinstance(params, np.ndarray):
            params = torch.from_numpy(params).float()
        return {
            'fc1.weight': params[ : self.hls[0]].reshape(self.hls[0], 1),
            'fc1.bias': params[self.hls[0] : 2 * self.hls[0]].reshape(self.hls[0]),
            'fc2.weight': params[2 * self.hls[0] : 2 * self.hls[0] + self.hls[0] * self.hls[1]].\
              reshape(self.hls[1], self.hls[0]),
            'fc2.bias': params[2 * self.hls[0] + self.hls[0] * self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + self.hls[1]].\
              reshape(self.hls[1]),
            'fc_mean.weight': params[2 * self.hls[0] + self.hls[0] * self.hls[1] + self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1]].\
              reshape(1, self.hls[1]),
            'fc_mean.bias': params[2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1] + 1].\
              reshape(1),
            'fc_std.weight': params[2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1] + 1 : 2 * self.hls[0] + self.hls[0] * self.hls[1] + 3 * self.hls[1] + 1].\
              reshape(1, self.hls[1]),
            'fc_std.bias': params[2 * self.hls[0] + self.hls[0] * self.hls[1] + 3 * self.hls[1] + 1 : ].\
              reshape(1),
        }


    def fitness_function_ea(self, i: int):
        # set local env and seed for every subprocesss
        local_env = deepcopy(self.env)
        torch.manual_seed(i)
        np.random.seed(i)

        state = local_env.reset()
        regret = 0
        for _ in range(self.cfgs.train.query_time):
            action = self.strategy(torch.FloatTensor(state)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = local_env.find_optim_action(action, idx=0)
            next_state, reward, _ = local_env.step_without_packing(action)
            regret += (opt_reward - reward[0]) / state[0]
            state = next_state

        regret /= self.cfgs.train.query_time
        return regret

    
    def fitness_function_interim(self, params):
        params = torch.from_numpy(params).float()
        new_s = Net(num_agents=1, num_actions=1, hidden_layer_size=self.hls).to(self.device)
        new_s.load_state_dict(self.reshape_params(params))
        
        regret_list = []
        for _ in range(self.cfgs.train.query_time):
            state = self.env.reset()
            regret, state0 = 0, state[0]
            for _ in range(self.cfgs.train.expect_time):
                action = new_s(torch.FloatTensor(state)\
                  .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
                _, opt_reward = self.env.find_optim_action(action, idx=0)
                next_state, reward, _ = self.env.step_without_packing(action)
                regret += (opt_reward - reward[0]) / state0
                state = next_state
                state[0], self.env.state[0] = state0, state0
            regret /= self.cfgs.train.expect_time
            regret_list.append(regret)

        return max(regret)

    
    def training(self):
        mp.set_start_method('spawn')
        init_params = [
            self.strategy.fc1.weight.data.numpy(), self.strategy.fc1.bias.data.numpy(),
            self.strategy.fc2.weight.data.numpy(), self.strategy.fc2.bias.data.numpy(),
            self.strategy.fc_mean.weight.data.numpy(), self.strategy.fc_mean.bias.data.numpy(),
            self.strategy.fc_std.weight.data.numpy(), self.strategy.fc_std.bias.data.numpy(),
        ]
        init_params_flat = np.concatenate([param.flatten() for param in init_params])

        es = CMAEvolutionStrategy(init_params_flat, 0.5, {'seed': self.cfgs.seed})
        best_params = es.ask()

        cnt = 0
        while not es.stop():
            cnt += 1
            st = time.time()
            print(f'update {cnt} >>> start time: {time.asctime(time.localtime(st))}')
            
            # CMA-ES
            solutions = [best_params[i] for i in range(es.popsize)]
            results = [None for _ in range(es.popsize)]
            for i, params in enumerate(solutions):
                self.strategy.load_state_dict(self.reshape_params(params))
                with mp.Pool(processes=mp.cpu_count()) as p:
                    result = p.map(self.fitness_function_ea, 
                              [j + i * mp.cpu_count() for j in range(mp.cpu_count())])
                    results[i] = sum(result) / len(result)
                    print(f'update {cnt} try {i} >>> regret: {results[i]}')

            # optimize and gen new params
            es.tell(solutions, results)
            best_params = es.ask()

            # save model
            best_idx = results.index(min(results))
            best_solution = solutions[best_idx]
            self.strategies.append(self.reshape_params(best_solution))
            torch.save(self.strategies, self.cfgs.path.model_path)

            et = time.time()
            # print(f'update {cnt} >>> end time: {time.asctime(time.localtime(et))}')
            print(f'update {cnt} >>> execute time: {(et - st) / 60} min')

        # best_solution = best_params[0]
        # torch.save(self.reshape_params(best_solution), self.cfgs.path.model_path)


if __name__ == '__main__':
    import yaml
    BASE_CONFIGS_PATH = r'./config/base.yaml'
    ES_CONFIGS_PATH = r'./config/es.yaml'

    with open(BASE_CONFIGS_PATH, 'r') as cfgs_file:
        base_cfgs = yaml.load(cfgs_file, Loader=yaml.FullLoader)
    base_cfgs = edict(base_cfgs)

    if ES_CONFIGS_PATH is not None:
        with open(ES_CONFIGS_PATH, 'r') as cfgs_file:
            cfgs = yaml.load(cfgs_file, Loader=yaml.FullLoader)
    else:
        cfgs = {}
    cfgs = edict(cfgs)
    cfgs.update(base_cfgs)

    trainer = Trainer(cfgs)
    trainer.strategy.load_state_dict(torch.load(cfgs.path.model_path))
    
