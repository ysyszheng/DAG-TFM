import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.DAGEnv import DAGEnv
from agents.es import Net
from utils.utils import log, fix_seed
from tqdm import tqdm
from easydict import EasyDict as edict
import csv
import torch
import gym
import concurrent.futures as cf
import logging
import time
import torch.multiprocessing as mp
from copy import deepcopy


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.strategies = Net(num_agents=1, num_actions=1, nu=self.cfgs.train.nu2).to(self.device)
        self.strategies.load_state_dict(torch.load(r'./models/es_0.pth')) # TODO: delete
        self.d = self.strategies.d
        self.J = int(4 + 3 * np.floor(np.log(self.d))) // 2 # 25 / 2

        self.env: DAGEnv = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )


    def OriginalMinusRegret(self, strategies: Net, alpha1=.01, nu1=.1):
        '''consider regret of agent 0
        original algorithm
        '''
        state = self.env.reset()

        # Oracle
        V = 0
        for _ in range(self.cfgs.train.inner_oracle_query_times):
            action = self.strategies(torch.FloatTensor(state).to(torch.float64)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            state, reward, _ = self.env.step_without_packing(action)
            V += reward[0]
        V /= self.cfgs.train.inner_oracle_query_times

        # NES
        for _ in range(self.cfgs.train.inner_iterations):
            epsilon = np.random.normal(0, 1, size=self.J)
            r = np.zeros(self.J)

            for j in range(self.J):
                for param in strategies.parameters():
                    param.data += nu1 * epsilon[j]
                
                # Oracle
                r_j_plus = 0
                for _ in range(self.cfgs.train.inner_oracle_query_times):
                    action0 = strategies(torch.FloatTensor(state[0].reshape(-1, 1)).to(torch.float64).to(self.device))\
                      .reshape(1).detach().cpu().numpy()
                    action = self.strategies(torch.FloatTensor(state[1:]).reshape(-1, 1).to(torch.float64).to(self.device))\
                      .squeeze().detach().cpu().numpy()
                    action = np.concatenate((action0, action))
                    state, reward, _ = self.env.step_without_packing(action)
                    r_j_plus += reward[0]
                r_j_plus /= self.cfgs.train.inner_oracle_query_times

                for param in strategies.parameters():
                    param.data -= 2 * nu1 * epsilon[j]

                # Oracle
                r_j_minus = 0
                for _ in range(self.cfgs.train.inner_oracle_query_times):
                    action0 = strategies(torch.FloatTensor(state[0].reshape(-1, 1)).to(torch.float64).to(self.device))\
                      .reshape(1).detach().cpu().numpy()
                    action = self.strategies(torch.FloatTensor(state[1:]).reshape(-1, 1).to(torch.float64).to(self.device))\
                      .squeeze().detach().cpu().numpy()
                    action = np.concatenate((action0, action))
                    state, reward, _ = self.env.step_without_packing(action)
                    r_j_minus += reward[0]
                r_j_minus /= self.cfgs.train.inner_oracle_query_times

                r[j] = (r_j_plus - r_j_minus)

                for param in strategies.parameters():
                    param.data += nu1 * epsilon[j]
            
            for param in strategies.parameters():
                param.data += alpha1 / (self.J * nu1) * sum(r * epsilon)

        # Oracle
        DEV = 0
        for _ in range(self.cfgs.train.inner_oracle_query_times):
            action0 = strategies(torch.FloatTensor(state[0].reshape(-1, 1)).to(torch.float64).to(self.device))\
              .reshape(1).detach().cpu().numpy()
            action = self.strategies(torch.FloatTensor(state[1:]).reshape(-1, 1).to(torch.float64).to(self.device))\
              .squeeze().detach().cpu().numpy()
            action = np.concatenate((action0, action))
            state, reward, _ = self.env.step_without_packing(action)
            DEV += reward[0]
        DEV /= self.cfgs.train.inner_oracle_query_times

        log(f'V: {V}, DEV: {DEV}')
        return V - DEV


    def MinusRegret(self, strategies: Net):
        """consider regret of agent 0
        directly use `env.find_optim_action()`
        """
        state = self.env.reset()

        regret = 0
        for _ in range(self.cfgs.train.inner_oracle_query_times):
            action = strategies(torch.FloatTensor(state).to(torch.float64)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = self.env.find_optim_action(action, idx=0)
            next_state, reward, _ = self.env.step_without_packing(action)
            regret += (opt_reward - reward[0]) / state[0]
            state = next_state

        regret /= self.cfgs.train.inner_oracle_query_times
        return -regret


    def MinusNashAPR(self, strategies: Net):
        """consider nash approximation
        directly use `env.find_all_optim_action()`
        """
        state = self.env.reset()

        nashapr = 0
        for _ in range(self.cfgs.train.apr_test_times):
            action = strategies(torch.FloatTensor(state).to(torch.float64)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = self.env.find_all_optim_action(action)
            next_state, reward, _ = self.env.step_without_packing(action)
            nashapr += np.amax((opt_reward - reward) / state)
            state = next_state

        nashapr /= self.cfgs.train.apr_test_times
        return -nashapr


    def MinusEpsilon(self, strategies: Net):
        """epsilon-BNE
        """
        state = self.env.reset()
        eps_list = []

        for _ in range(self.cfgs.train.inner_oracle_query_times):
            eps, state_0 = 0, state[0]
            for _ in range(self.cfgs.train.expect_time):
                self.env.state[0], state[0] = state_0, state_0
                action = strategies(torch.FloatTensor(state).to(torch.float64)\
                    .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
                _, opt_reward = self.env.find_optim_action(action, idx=0)
                state, reward, _ = self.env.step_without_packing(action)
                eps += (opt_reward - reward[0]) / state_0
            eps /= self.cfgs.train.expect_time
            eps_list.append(eps)

        eps = np.amax(eps_list)
        return -eps


    def EPSworker(self, strategies: Net):
        env = deepcopy(self.env)
        state = env.reset()
        state_0 = state[0]
        epsilon = 0

        for _ in range(self.cfgs.train.expect_time):
            env.state[0], state[0] = state_0, state_0
            action = strategies(torch.FloatTensor(state).to(torch.float64)\
                .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = env.find_optim_action(action, idx=0)
            state, reward, _ = env.step_without_packing(action)
            epsilon += (opt_reward - reward[0]) / state_0

        epsilon /= self.cfgs.train.expect_time
        return epsilon


    def NESworker(self, j, nu2, epsilon):
        """Natural Evolution Strategies
        """
        strategies_copy = Net(num_agents=1, num_actions=1).to(self.device)
        strategies_copy.load_state_dict(self.strategies.state_dict())

        with torch.no_grad():
            for param, delta in zip(strategies_copy.parameters(), nu2 * epsilon):
                param.data.add_(delta)

        r_j = self.MinusRegret(strategies_copy) # minus regret of agent 0

        return j, r_j

    
    def MiniMax(self, alpha1, nu1, alpha2, nu2, beta1=.9, beta2=.999, eps=1e-8):
        """min regret
        use adam to optimize lr alpha2
        """
        strategies_new = Net(num_agents=1, num_actions=1).to(self.device)
        for t in range(self.cfgs.train.outer_iterations):
            print(f'************* iter: {t} *************')
            start_time = time.time()
            print(f'start time: {time.asctime(time.localtime(start_time))}')

            # before update
            with mp.Pool(processes=self.cfgs.train.inner_oracle_query_times) as p:
            # with mp.Pool(processes=mp.cpu_count()) as p:
                results = p.starmap(self.EPSworker, [(self.strategies,)
                                for _ in range(self.cfgs.train.inner_oracle_query_times)])
                print(f'update time {t}>>> epsilon: {np.amax(results)}')

            epsilon = np.random.normal(0, 1, size=(self.J, self.d))
            epsilon = np.concatenate((epsilon, -epsilon), axis=0)
            u = np.zeros(2 * self.J)
            r = np.zeros(2 * self.J)

            # max epsilon-BNE
            for j in range(2 * self.J):
                strategies_new.load_state_dict(self.strategies.state_dict())
                with torch.no_grad():
                    for param, delta in zip(strategies_new.parameters(), nu2 * epsilon[j]):
                        param.data.add_(delta)

                with mp.Pool(processes=self.cfgs.train.inner_oracle_query_times) as p:
                    results = p.starmap(self.EPSworker, [(strategies_new,)
                                    for _ in range(self.cfgs.train.inner_oracle_query_times)])
                    r[j] = -np.amax(results)
                    # r[j] = -np.mean(results)
                    print(f'try {j}>>> epsilon: {np.amax(results)}, regret: {np.mean(results)}')
            
            # fitness shaping
            for k in range(1, 2 * self.J + 1):
                u[k - 1] = max(0, np.log(self.J + 1) - np.log(k))
            u = u / np.sum(u) - 1 / (2 * self.J)

            sorted_indices = sorted(range(len(r)), key=lambda i: r[i], reverse=True)
            gradient = np.sum([u[k] * epsilon[sorted_indices[k]] for k in range(2 * self.J)], axis=0) / nu2

            # non fitness shaping
            # gradient = np.sum([r[k] * epsilon[k] for k in range(2 * self.J)], axis=0) / (nu2 * self.J)

            # Adam optimize
            self.strategies.update(gradient, alpha2, (beta1, beta2), eps)

            # save network param & adam param
            torch.save(self.strategies.state_dict(), self.cfgs.path.model_path)

            end_time = time.time()
            print(f'end time: {time.asctime(time.localtime(end_time))}')
            print(f'execute time: {(end_time - start_time) / 60} min')


    def training(self):
        mp.set_start_method('spawn')
        self.MiniMax(self.cfgs.train.alpha1, self.cfgs.train.nu1, self.cfgs.train.alpha2, self.cfgs.train.nu2)
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
