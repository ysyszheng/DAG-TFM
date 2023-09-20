from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import gym
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve, root_scalar
from gym import spaces
import numpy as np
import random
from utils.utils import log


class DAGEnv(gym.Env):
    def __init__(self,
                 fee_data_path,
                 max_agents_num,
                 lambd,
                 delta,
                 a,
                 b,
                 is_burn=False):
        self.max_agents_num = max_agents_num
        self.lambd = lambd
        self.delta = delta
        self.a = a
        self.b = b
        self.is_burn = is_burn
        self.eps = 1e-8

        # * action: transaction fee
        self.action_space = spaces.Box(
            low=0, high=np.inf, shape=(max_agents_num,), dtype=np.float32)
        # * state: private values of agents and number of agents
        # * let numbers of agents be fixed, i.e. num_agents = max_agents_num
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(max_agents_num,), dtype=np.float32)

        self.fee_list = np.load(fee_data_path)
        self.state_mean = np.mean(self.fee_list)
        self.state_std = np.std(self.fee_list)

    def f(self, p):
        assert np.all(p >= 0 and p <= 1)
        if np.isscalar(p):
            if p == 0:
                return 1
            else:
                return (1 - np.exp(-self.lambd * self.delta * p)) / (self.lambd * self.delta * p)
        else:
            ratios = np.ones_like(p)
            mask = (p != 0)

            ratios[mask] = (1 - np.exp(-self.lambd * self.delta * p[mask])) / (self.lambd * self.delta * p[mask])
            return ratios

    def invf(self, y):
        # ! Don't use this function, use invf_with_clip instead
        # * bracket should be [0, np.inf]
        if np.isscalar(y):
            result = root_scalar(lambda x: self.f(x) - y, method='brentq', bracket=[0, 1])
            return result.root
        else:
            roots = np.zeros_like(y)
            for i, y_val in np.ndenumerate(y):
                result = root_scalar(lambda x: self.f(x) - y_val, method='brentq', bracket=[0, 1])
                roots[i] = result.root
            return roots
        
    def invf_with_clip(self, y):
        assert np.all(y >= 0 and y <= 1)
        if np.isscalar(y):
            if y <= self.f(1):
                return 1
            else:
                result = root_scalar(lambda x: self.f(x) - y, method='brentq', bracket=[0, 1])
                return result.root
        else:
            roots = np.zeros_like(y)
            for i, y_val in np.ndenumerate(y):
                if y_val <= self.f(1):
                    roots[i] = 1
                else:
                    result = root_scalar(lambda x: self.f(x) - y_val, method='brentq', bracket=[0, 1])
                    roots[i] = result.root
            return roots

    def reset(self):
        # self.num_agents = np.random.randint(self.max_agents_num)
        # * fix the number of agents
        self.num_agents = self.max_agents_num
        self.state = np.array([self.fee_list[i] for i in np.random.randint(len(self.fee_list), size=self.num_agents)])
        self.num_miners = 1 + np.random.poisson(self.lambd)

        return self.state

    def step(self, actions):
        rewards, probabilities = self.calculate_rewards(actions)
        done = True
        self.reset()

        return self.state, rewards, done, {"probabilities": probabilities}

    def find_optim_action(self, actions, idx=0, cnt=100):
        assert idx >= 0 and idx < self.num_agents
        s = self.state[idx]
        max_reward = -1
        optim_action = 0
        for a in np.linspace(0, s, cnt):
            actions[idx] = a
            rewards, _ = self.calculate_rewards(actions)
            print(a, ',', rewards[idx])
            if rewards[idx] > max_reward:
                max_reward = rewards[idx]
                optim_action = a
        return optim_action

    def calculate_probabilities(self, actions: np.ndarray) -> np.ndarray:
        # * probability of being included in the block
        # * if action <= 0, then probability = 0
        zero_indices = np.where(actions <= 0)[0]
        actions_without_zero = np.delete(actions, zero_indices)

        if len(actions_without_zero) <= self.b:
            probabilities = np.ones(self.num_agents)
            probabilities[zero_indices] = 0
            return probabilities
        
        if np.all(actions_without_zero == actions_without_zero[0]):
            probabilities = np.ones(self.num_agents) * (self.b / len(actions_without_zero))
            probabilities[zero_indices] = 0
            return probabilities

        probabilities = np.zeros(self.num_agents)
        v, k = np.unique(actions_without_zero, return_counts=True)
        v, k = v[::-1], k[::-1]

        def G(l, z):
            assert l >= 0 and l < self.num_agents
            res = 0
            for h in range(l+1):
                res += k[h] * self.invf_with_clip(z/(v[h]))
            return (res - self.b)

        k_max = -1
        flag = False
        while (k_max + 1) < len(v) and not flag:
            k_max += 1
            for l in range(k_max+1):
                if G(l, v[l]) > 0:
                    flag = True
                    k_max -= 1
                    break

        def G_k_max(z): return G(k_max, z)

        # self.plot(G_k_max, (0, v[k_max]))
        # log('b', self.b, 'num_agents', self.num_agents)
        # log('G(0, v[0])', G(0, v[0]))
        # log('actions_without_zero', actions_without_zero)
        # log('v', v)
        # log('k', k)
        # log('k_max', k_max, 'v[k_max]', v[k_max])
        # log('G_k_max(0)', G_k_max(0), 'G_k_max(v[k_max])', G_k_max(v[k_max]))
        
        c_k_max = brentq(G_k_max, 0, v[k_max])
        # c_k_max = fsolve(G_k_max, x0=v[k_max])[0]

        p = [0 for _ in range(self.num_agents)]
        for l in range(k_max+1):  # k[l] != 0
            p[l] = self.invf_with_clip(c_k_max/(v[l]))

        for i in range(self.num_agents):
            if actions[i] <= 0:
                probabilities[i] = 0
            else:
                idx = np.where(actions[i] == v)[0][0]
                probabilities[i] = p[idx]
        return probabilities

    def calculate_rewards(self, actions):
        rewards = np.zeros(self.num_agents)

        if self.is_burn:
            actions = self.a * np.log(1 + actions / self.a)
            actions[np.isnan(actions)] = 0

        probabilities = self.calculate_probabilities(actions)

        for i in range(self.num_agents):
            private_value = self.state[i]
            offer_price = actions[i]
            probability = probabilities[i]

            rewards[i] =  (1 - (1 - probability) ** self.num_miners) * \
                (private_value - offer_price)

        return rewards, probabilities

    def plot(self, func=None, bounds=(-5, 5), num=1000):
        x = np.linspace(bounds[0], bounds[1], num)
        if func is not None:
            y = func(x)
        plt.plot(x, y)
        plt.title(func.__name__)
        plt.show()


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    BASE_CONFIGS_PATH = r'./config/base.yaml'

    with open(BASE_CONFIGS_PATH, 'r') as cfg_file:
        base_cfgs = yaml.load(cfg_file, Loader=yaml.FullLoader)
    base_cfgs = edict(base_cfgs)

    env = DAGEnv(
            fee_data_path=base_cfgs.fee_data_path,
            max_agents_num=base_cfgs.max_agents_num,
            lambd=base_cfgs.lambd,
            delta=base_cfgs.delta,
            b=base_cfgs.b,
            is_burn=base_cfgs.is_burn
    )
    env.plot(env.f, (0, 1))
    env.plot(env.invf, (env.f(1), 1))
    env.plot(env.invf_with_clip, (0, 1))
