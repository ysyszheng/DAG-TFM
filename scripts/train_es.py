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
        if self.cfgs.train.is_load:
            self.strategies.load_state_dict(torch.load(self.cfgs.path.model_path))

        self.d = self.strategies.d
        self.J = int(4 + 3 * np.floor(np.log(self.d))) // 2 # 25 / 2

        self.env: DAGEnv = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.path.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, burn_flag=cfgs.burn_flag,
            clip_value=cfgs.clip_value, norm_value=cfgs.norm_value, log_file_path=cfgs.path.log_path,
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
        logger = init_logger(f'worker {i}', self.cfgs.path.log_path)
        fix_seed(seed)
        logger.debug(f'start')

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
        
        regret = np.array(regret)
        mean_ = sum(regret) / len(regret) # TODO: only consider non-zero value ?
        max_ = max(regret)
        logger.debug(f'end, mean regret: {mean_}, max regret: {max_}')

        return i, -mean_


    def eval_(self):
        logger = init_logger('worker eval', self.cfgs.path.log_path)
        logger.debug(f'start')
        fix_seed(65535)

        env = deepcopy(self.env)
        regret = []

        for _ in range(self.cfgs.train.expect_time):
            state = env.reset()
            action = self.strategies(torch.FloatTensor(state).to(torch.float64)\
                .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = env.find_optim_action(action, idx=0)
            _, reward, _ = env.step_without_packing(action)
            regret.append((opt_reward - reward[0]) / state[0])

        regret = np.array(regret)
        mean_ = sum(regret) / len(regret)
        max_ = max(regret)
        logger.debug(f'end, mean regret: {mean_}, max regret: {max_}')

        return mean_, max_
    

    def MiniMax(self, alpha2, beta1=.9, beta2=.999, eps=1e-8):
        """min regret
        use adam to optimize lr alpha2
        """
        torch.set_grad_enabled(False)

        for t in range(self.cfgs.train.outer_iterations):
            start_time = time.time()

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
            self.logger.debug(f'update time {t + 1}, min regret: {min(-r)}, execute time: {(end_time - start_time) / 60} min')
    
        torch.set_grad_enabled(True)


    def training(self):
        self.logger.info(f'========== {self.cfgs.method} Trainer start, lambda: {self.cfgs.lambd}, burn flag: {self.cfgs.burn_flag}, a: {self.cfgs.a} ==========')
        mp.set_start_method('spawn')
        self.MiniMax(self.cfgs.train.alpha2)
        self.logger.info(f'========== {self.cfgs.method} Trainer end, lambda: {self.cfgs.lambd}, burn flag: {self.cfgs.burn_flag}, a: {self.cfgs.a} ==========\n')


    def plot_strategy(self, size=20, alpha=.5):
        fix_seed(int(time.time())) # TODO:

        if self.cfgs.burn_flag == 'non':
            title = f'y=x'
        elif self.cfgs.burn_flag == 'log':
            title = f'y={self.cfgs.a}*log(1+x/{self.cfgs.a})'
        elif self.cfgs.burn_flag == 'poly':
            title = f'y=x^{self.cfgs.a}'
        if self.cfgs.norm_value is not None:
            unit = f'{self.cfgs.norm_value} SATs'
        else:
            unit = f'SATs'

        state = self.env.reset(miner_mode='fix')
        title = f'burning rule:{title}, delay {self.env.delta} s, expected #miner: {self.env.delta * self.env.lambd}, true #miner: {self.env.num_miners}'

        self.strategies.load_state_dict(torch.load(self.cfgs.path.model_path))
        action = self.strategies(
            torch.FloatTensor(state).to(torch.float64).reshape(-1, 1).to(self.device)
        ).squeeze().detach().cpu().numpy()
        
        opt_action, opt_reward = self.env.find_all_optim_action(action)
        _, reward, _, info = self.env.step(action)
        regret = (opt_reward-reward)/state

        title = f'{title}, throughput: {info["throughput"]} tps'

        self.logger.debug(f'state: {state}')
        self.logger.debug(f'action: {action}')
        self.logger.debug(f'expected reward: {reward}')
        self.logger.debug(f"probs: {info['probabilities']}")
        self.logger.debug(f"include: {info['included_txs']}")
        self.logger.debug(f"true reward: {info['true_reward']}")

        self.logger.info(f'========== regret info ==========')
        self.logger.info(f'mean regret: {sum(regret)/len(regret)}, max regret: {max(regret)}')
        self.logger.info(f'========== regret info ==========\n')

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=200)

        ax1.plot(state, state, label='Truthful')
        ax1.scatter(state, action, s=size, marker='o', alpha=alpha, label='Bid')
        ax1.scatter(state, opt_action, s=size, marker='o', alpha=alpha, label='Optimal Bid')
        ax1.scatter(state, reward, s=size, marker='^', alpha=alpha, label='Expected Reward')
        ax1.scatter(state, info['true_reward'], marker='^', s=size, alpha=alpha, label='True Reward')
        ax1.scatter(state, opt_reward, marker='^', s=size, alpha=alpha, label='Optimal Reward')
        ax1.set_xlabel(f'Valuation / {unit}')
        ax1.set_ylabel(f'Bid or Reward / {unit}')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True)
        
        ax2.scatter(state, info['probabilities'], marker='o', s=size, alpha=alpha, label='Probabilities')
        ax2.scatter(state, info['included_txs'], marker='^', s=size, alpha=alpha, label='Is included')
        ax2.scatter(state, regret, marker='+', s=size, alpha=alpha, label='regret=(E[r]-max r)/v')
        ax2.set_xlabel(f'Valuation / {unit}')
        ax2.set_ylabel('Probability or Regret')
        ax2.legend()
        ax2.grid(True)

        # ax3.scatter(action, info['probabilities'], marker='o', s=size, alpha=alpha, label='Probabilities')
        # ax3.set_xlabel(f'Bid / {self.cfgs.norm_value} SATs')
        # ax3.set_ylabel('Probability')
        # ax3.legend()
        # ax3.grid(True)

        plt.tight_layout()

        plt.savefig(self.cfgs.path.img_path)
        plt.show()
        self.logger.info(f'========== save figure ==========')
        self.logger.info(f'save figure to {self.cfgs.path.img_path}')
        self.logger.info(f'========== save figure ==========\n')


if __name__ == '__main__':
    pass
