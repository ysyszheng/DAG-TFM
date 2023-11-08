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


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )

        self.strategies = Net(num_agents=1, num_actions=1).to(self.device)
        
        total_params = sum(p.numel() for p in self.strategies.parameters())
        self.J = int(4 + 3 * np.floor(np.log(total_params))) # 25


    def OriginalMinusRegret(self, strategies: Net, alpha1=.01, mu1=.1):
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
                    param.data += mu1 * epsilon[j]
                
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
                    param.data -= 2 * mu1 * epsilon[j]

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
                    param.data += mu1 * epsilon[j]
            
            for param in strategies.parameters():
                param.data += alpha1 / (self.J * mu1) * sum(r * epsilon)

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


    def NESworker(self, j, mu2, epsilon, flag: bool):
        """Natural Evolution Strategies
        """
        strategies_copy = Net(num_agents=1, num_actions=1).to(self.device)
        strategies_copy.load_state_dict(self.strategies.state_dict())

        for param in strategies_copy.parameters():
            param.data += mu2 * epsilon[j] if flag else -(mu2 * epsilon[j])

        r_j_flag = self.MinusRegret(strategies_copy) # minus regret of agent 0
        # r_j_flag = self.MinusNashAPR(strategies_copy) # minus nash approximation

        print(f'worker {j}{"+" if flag else "-"} end, r[{j}]{"+" if flag else "-"}: {r_j_flag}, epsilon[{j}]: {epsilon[j]}')
        return j, r_j_flag, flag

    
    def MiniMax(self, alpha1, mu1, alpha2, mu2, beta1=.9, beta2=.999, eps=1e-8):
        """min regret
        use adam to optimize lr alpha2
        """
        m, v = 0, 0
        for t in range(self.cfgs.train.outer_iterations):
            print(f'************* iter: {t} *************')
            start_time = time.time()
            print(f'start time: {time.asctime(time.localtime(start_time))}')

            epsilon = np.random.normal(0, 1, size=self.J)
            r_plus = np.zeros(self.J)
            r_minus = np.zeros(self.J)
            r = np.zeros(self.J)

            # multi process
            iter_args = []
            for j in range(self.J):
                iter_args.append((j, mu2, epsilon, True))
                iter_args.append((j, mu2, epsilon, False))

            with mp.Pool(processes=mp.cpu_count()) as p:
                regret = p.apply_async(self.MinusRegret, (self.strategies,))
                results = p.starmap(self.NESworker, iter_args)

                print(f'!!!!!! regert (before update in iter {t}): {-regret.get()}')
                for j, r_j_flag, flag in results:
                    if flag:
                        r_plus[j] = r_j_flag
                    else:
                        r_minus[j] = r_j_flag
                r = r_plus - r_minus

            # Adam
            print(r)
            print(epsilon)
            print(self.J, mu2)
            gradient = np.sum(r * epsilon) / (self.J * mu2)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat = m / (1 - beta1 ** (t + 1))
            v_hat = v / (1 - beta2 ** (t + 1))

            delta_param = alpha2 * m_hat / (np.sqrt(v_hat) + eps)
            print(f'gradient: {gradient}, delta param: {delta_param}')

            for param in self.strategies.parameters():
                param.data += delta_param

            torch.save(self.strategies.state_dict(), self.cfgs.path.model_path)

            end_time = time.time()
            print(f'end time: {time.asctime(time.localtime(end_time))}')
            print(f'execute time: {(end_time - start_time) / 60} min')

        # return self.MinusRegret()
                

    def training(self):
        fix_seed(self.cfgs.seed)
        mp.set_start_method('spawn')
        self.MiniMax(self.cfgs.train.alpha1, self.cfgs.train.mu1, self.cfgs.train.alpha2, self.cfgs.train.mu2)
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
