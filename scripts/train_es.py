import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from envs.DAGEnv import DAGEnv
from agents.es import Net
from utils.utils import log, fix_seed, init_logger
from tqdm import tqdm
from easydict import EasyDict as edict
import csv
import torch
import gym
import concurrent.futures as cf
import time
import torch.multiprocessing as mp
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.strategies = Net(num_agents=1, num_actions=1, nu=self.cfgs.train.nu2).to(self.device)
        self.d = self.strategies.d
        self.J = int(4 + 3 * np.floor(np.log(self.d))) // 2 # 25 / 2

        self.env: DAGEnv = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
            sats_to_btc=cfgs.sats_to_btc,
        )

        self.logger = init_logger(__name__, self.cfgs.path.log_path)


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
    

    def TESTworker(self, strategies: Net):
        x = np.random.random(1000) * 10000
        y_hat = strategies(torch.FloatTensor(x).to(torch.float64)\
                .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
        y = np.sqrt(x)
        loss = np.max(np.abs(y - y_hat))

        return -loss, x, y_hat, y


    def NESworker(self, e, i, seed):
        """Natural Evolution Strategies
        """
        logger = init_logger(__name__, self.cfgs.path.log_path)
        fix_seed(seed)
        logger.debug(f'worker {i} start')

        snet = Net(num_agents=1, num_actions=1).to(self.device)
        snet.load_state_dict(self.strategies.state_dict())
        with torch.no_grad():
            for param, delta in zip(snet.parameters(), snet.nu * e):
                param.data.add_(delta)

        env = deepcopy(self.env)

        regret = []
        for _ in range(self.cfgs.train.expect_time):
            state = env.reset()
            action = snet(torch.FloatTensor(state).to(torch.float64)\
                .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = env.find_optim_action(action, idx=0)
            _, reward, _ = env.step_without_packing(action)
            regret.append((opt_reward - reward[0]) / state[0])

        mean_ = sum(regret) / len(regret)
        max_ = max(regret)
        logger.debug(f'worker {i} end, mean regret: {mean_}, max regret: {max_}, regret: {regret}')

        return i, -mean_ # TODO!


    def eval_(self):
        logger = init_logger(__name__, self.cfgs.path.log_path)
        logger.debug(f'worker eval start')
        # fix_seed(-1)
        regret = []
        env = deepcopy(self.env)

        for _ in range(self.cfgs.train.expect_time):
            state = env.reset()
            action = self.strategies(torch.FloatTensor(state).to(torch.float64)\
                .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = env.find_optim_action(action, idx=0)
            _, reward, _ = env.step_without_packing(action)
            regret.append((opt_reward - reward[0]) / state[0])

        mean_ = sum(regret) / len(regret)
        max_ = max(regret)
        logger.debug(f'worker eval end, regret: {regret}')
        return mean_, max_
    

    def MiniMax(self, alpha2, beta1=.9, beta2=.999, eps=1e-8):
        """min regret
        use adam to optimize lr alpha2
        """
        torch.set_grad_enabled(False)

        for t in range(self.cfgs.train.outer_iterations):
            # print(f'************* iter: {t} *************')
            start_time = time.time()
            # print(f'start time: {time.asctime(time.localtime(start_time))}')

            epsilon = np.random.normal(0, 1, size=(self.J, self.d))
            epsilon = np.concatenate((epsilon, -epsilon), axis=0)
            u = np.zeros(2 * self.J)
            r = np.zeros(2 * self.J)

            with mp.Pool(processes=mp.cpu_count()) as p:
                eval_result = p.apply_async(self.eval_)
                results = p.starmap(self.NESworker, [(epsilon[i], i, t * 2 * self.J + i) for i in range(2 * self.J)])

                eval_mean, eval_max = eval_result.get()
                self.logger.critical(f'update time {t}, mean regret: {eval_mean}, max regret: {eval_max}')

                for result in results:
                    r[result[0]] = result[1]

            # fitness shaping
            for k in range(1, 2 * self.J + 1):
                u[k - 1] = max(0, np.log(self.J + 1) - np.log(k))
            u = u / np.sum(u) - 1 / (2 * self.J)

            sorted_indices = sorted(range(len(r)), key=lambda i: r[i], reverse=True)

            # # calculate gradient
            # grad_log_pi = [np.concatenate((
            #         epsilon[k] / self.strategies.nu,
            #         (epsilon[k] ** 2 - 1) / self.strategies.nu
            #     ), axis=None) for k in range(2 * self.J)]
            # gradient = np.sum([u[k] * grad_log_pi[sorted_indices[k]] for k in range(2 * self.J)], axis=0)
            
            # # fisher matrix and natural gradient
            # F = np.sum([grad_log_pi[k].reshape(-1,1) @ grad_log_pi[k].reshape(-1,1).T 
            #             for k in range(2 * self.J)], axis=0) / (2 * self.J)
            # inv_F = np.linalg.inv(F)
            # inv_F /= np.amax(inv_F)
            # natural_gradient = inv_F @ gradient

            gradient = np.sum([u[k] * epsilon[sorted_indices[k]] for k in range(2 * self.J)], axis=0) / self.strategies.nu

            # Adam optimize
            self.strategies.update(gradient, alpha2, (beta1, beta2), eps)

            # save network param & adam param
            torch.save(self.strategies.state_dict(), self.cfgs.path.model_path)
            self.logger.debug(f'save model to {self.cfgs.path.model_path}')

            end_time = time.time()
            # print(f'end time: {time.asctime(time.localtime(end_time))}')
            self.logger.debug(f'update time {t + 1}, min regret: {min(-r)}, execute time: {(end_time - start_time) / 60} min')
    
        torch.set_grad_enabled(True)


    def training(self):
        self.logger.info(f'*** NES Trainer start, lambda: {self.cfgs.lambd}, is_burn: {self.cfgs.burn_flag}, a: {self.cfgs.a}')
        mp.set_start_method('spawn')
        self.MiniMax(self.cfgs.train.alpha2)
        self.logger.info(f'*** NES Trainer end, lambda: {self.cfgs.lambd}, is_burn: {self.cfgs.burn_flag}, a: {self.cfgs.a}')


    def plot_strategy(self):
        self.strategies.load_state_dict(torch.load(self.cfgs.path.model_path))

        x = env.reset()
        y = self.strategies(torch.FloatTensor(x).to(torch.float64)\
            .reshape(-1, 1).to(device)).squeeze().detach().cpu().numpy()

        plt.figure()
        plt.plot(x, x, color='red', label='Truthful')
        plt.scatter(x, y, s=5, alpha=.8, color='blue', label='Strategy')
        plt.title('Fee - Valuation')
        plt.xlabel('Valuation')
        plt.ylabel('Transaction Fee')
        plt.legend()
        plt.savefig(self.cfgs.path.img_path)
        plt.show()


if __name__ == '__main__':
    import yaml
    import argparse

    BASE_CONFIGS_PATH = r'./config/base.yaml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=None, help='')
    parser.add_argument('--is_burn', type=int, default=None, help='')
    parser.add_argument('--a', type=float, default=None, help='')
    args = parser.parse_args()

    with open(BASE_CONFIGS_PATH, 'r') as cfgs_file:
        base_cfgs = yaml.load(cfgs_file, Loader=yaml.FullLoader)
    base_cfgs = edict(base_cfgs)

    env: DAGEnv = gym.make('gym_dag_env-v0', 
        fee_data_path=base_cfgs.fee_data_path, is_clip=base_cfgs.is_clip, 
        clip_value=base_cfgs.clip_value, max_agents_num=base_cfgs.max_agents_num,
        lambd=args.lambd, delta=base_cfgs.delta, a=args.a, b=base_cfgs.b, is_burn=args.is_burn,
        sats_to_btc=base_cfgs.sats_to_btc, seed=int(time.time())
    )

    burn_flag = ['no', 'log', 'poly']
    args.a = None if args.is_burn == 0 else args.a
    fn = f'ES_{args.lambd}_{burn_flag[args.is_burn]}_{args.a}'

    device = torch.device('cpu')
    strategies = Net(num_agents=1, num_actions=1).to(device)
    strategies.load_state_dict(torch.load(f'./results/models/{fn}.pth'))

    x = env.reset()
    y = strategies(torch.FloatTensor(x).to(torch.float64)\
        .reshape(-1, 1).to(device)).squeeze().detach().cpu().numpy()

    plt.figure()
    plt.plot(x, x, color='red', label='Truthful')
    plt.scatter(x, y, s=5, alpha=.8, color='blue', label='Strategy')
    plt.title('Fee - Valuation')
    plt.xlabel('Valuation')
    plt.ylabel('Transaction Fee')
    plt.legend()
    plt.savefig(f'./results/img/{fn}.png')
    plt.show()
