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


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        self.cfgs.train.hidden_layer_size = ast.literal_eval(self.cfgs.train.hidden_layer_size)
        self.hls = self.cfgs.train.hidden_layer_size
        fix_seed(cfgs.seed)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.strategies = Net(num_agents=1, num_actions=1, hidden_layer_size=self.hls).to(self.device)

        self.env: DAGEnv = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )


    def fitness_function(self, params):
        params = torch.from_numpy(params).float()
        new_s = Net(num_agents=1, num_actions=1, hidden_layer_size=self.hls).to(self.device)
        new_s.load_state_dict({
            'fc1.weight': params[ : self.hls[0]].reshape(self.hls[0], 1),
            'fc1.bias': params[self.hls[0] : 2 * self.hls[0]].reshape(self.hls[0]),
            'fc2.weight': params[2 * self.hls[0] : 2 * self.hls[0] + self.hls[0] * self.hls[1]].\
              reshape(self.hls[1], self.hls[0]),
            'fc2.bias': params[2 * self.hls[0] + self.hls[0] * self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + self.hls[1]].\
              reshape(self.hls[1]),
            'fc3.weight': params[2 * self.hls[0] + self.hls[0] * self.hls[1] + self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1]].\
              reshape(1, self.hls[1]),
            'fc3.bias': params[2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1] : ].\
              reshape(1)
        })
        
        state = self.env.reset()

        regret = 0
        for _ in range(self.cfgs.train.query_time):
            action = new_s(torch.FloatTensor(state)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = self.env.find_optim_action(action, idx=0)
            next_state, reward, _ = self.env.step_without_packing(action)
            regret += (opt_reward - reward[0]) / state[0]
            state = next_state

        regret /= self.cfgs.train.query_time
        return regret

        # x = np.random.random(10000) * 500
        # y_hat = new_s(torch.FloatTensor(x)\
        #         .reshape(-1, 1).to(self.device)).squeeze().detach().cpu()
        # y = np.sqrt(x)
        # mse = torch.mean(torch.square(y_hat - torch.from_numpy(y).float()))
        # return mse.item()

    
    def training(self):
        mp.set_start_method('spawn')
        init_params = [
            self.strategies.fc1.weight.data.numpy(), self.strategies.fc1.bias.data.numpy(),
            self.strategies.fc2.weight.data.numpy(), self.strategies.fc2.bias.data.numpy(),
            self.strategies.fc3.weight.data.numpy(), self.strategies.fc3.bias.data.numpy(),
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
            with mp.Pool(processes=mp.cpu_count()) as p:
                results = p.map(self.fitness_function, solutions)
                es.tell(solutions, results)
                print(f'update {cnt} >>> {results}')
            best_params = es.ask()

            # save model
            best_idx = results.index(min(results))
            best_solution = solutions[best_idx]
            best_solution = torch.from_numpy(best_solution).float()
            self.strategies.load_state_dict({
                'fc1.weight': best_solution[ : self.hls[0]].reshape(self.hls[0], 1),
                'fc1.bias': best_solution[self.hls[0] : 2 * self.hls[0]].reshape(self.hls[0]),
                'fc2.weight': best_solution[2 * self.hls[0] : 2 * self.hls[0] + self.hls[0] * self.hls[1]].\
                  reshape(self.hls[1], self.hls[0]),
                'fc2.bias': best_solution[2 * self.hls[0] + self.hls[0] * self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + self.hls[1]].\
                  reshape(self.hls[1]),
                'fc3.weight': best_solution[2 * self.hls[0] + self.hls[0] * self.hls[1] + self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1]].\
                  reshape(1, self.hls[1]),
                'fc3.bias': best_solution[2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1] : ].\
                  reshape(1)
            })
            torch.save(self.strategies.state_dict(), self.cfgs.path.model_path)

            et = time.time()
            print(f'update {cnt} >>> end time: {time.asctime(time.localtime(et))}')
            print(f'update {cnt} >>> execute time: {(et - st) / 60} min')

        best_solution = best_params[0]
        best_solution = torch.from_numpy(best_solution).float()
        self.strategies.load_state_dict({
            'fc1.weight': best_solution[ : self.hls[0]].reshape(self.hls[0], 1),
            'fc1.bias': best_solution[self.hls[0] : 2 * self.hls[0]].reshape(self.hls[0]),
            'fc2.weight': best_solution[2 * self.hls[0] : 2 * self.hls[0] + self.hls[0] * self.hls[1]].\
              reshape(self.hls[1], self.hls[0]),
            'fc2.bias': best_solution[2 * self.hls[0] + self.hls[0] * self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + self.hls[1]].\
              reshape(self.hls[1]),
            'fc3.weight': best_solution[2 * self.hls[0] + self.hls[0] * self.hls[1] + self.hls[1] : 2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1]].\
              reshape(1, self.hls[1]),
            'fc3.bias': best_solution[2 * self.hls[0] + self.hls[0] * self.hls[1] + 2 * self.hls[1] : ].\
              reshape(1)
        })

        torch.save(self.strategies.state_dict(), self.cfgs.path.model_path)


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
    trainer.strategies.load_state_dict(torch.load(cfgs.path.model_path))
    print(f'regert: {-trainer.MinusRegret(trainer.strategies)}')
