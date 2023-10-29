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
from utils.utils import log, fix_seed, get_dist_from_margin


class DAGEnv(gym.Env):
    def __init__(self,
                 fee_data_path,
                 is_clip,
                 clip_value,
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
        self.is_clip = is_clip
        self.clip_value = clip_value

        # action: transaction fee
        self.action_space = spaces.Box(
            low=0, high=np.inf, shape=(max_agents_num,), dtype=np.float32)
        # state: private values of agents and 
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(max_agents_num,), dtype=np.float32)

        self.fee_list = np.load(fee_data_path)
        if self.is_clip:
            self.fee_list = self.fee_list[self.fee_list <= self.clip_value]
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
        # bracket should be [0, np.inf]
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
        # fix the number of txs in delta time (fix number of agents)
        self.num_agents = self.max_agents_num
        self.state = np.array([self.fee_list[i] for i in np.random.randint(len(self.fee_list), size=self.num_agents)])
        self.num_miners = 1 + np.random.poisson(self.lambd * self.delta)

        return self.state

    def step(self, actions):
        # calculate rewards and probabilities
        rewards, probabilities = self.calculate_rewards(actions)

        # calculate throughput and social welfare (total private value)
        included_txs = []
        # use marginal probabilities
        # for _ in range(self.num_miners):
        #     random_numbers = np.random.rand(self.num_agents)
        #     mask = random_numbers < probabilities
        #     selected_indices = np.where(mask)[0]
        #     included_txs.extend(selected_indices)

        threshold = 1e-5
        if abs(sum(probabilities) - self.b) <= threshold:
            for _ in range(self.num_miners):
                event_set, p = get_dist_from_margin(self.b, self.num_agents, probabilities.tolist())
                print(sum(p))
                p = p / sum(p) # sum(p) approx 1
                txs = list(random.choices(event_set, p, k=1)[0])
                included_txs.append(txs)
        elif sum(probabilities) < self.b:
            for _ in range(self.num_miners):
                zero_indices = np.where(probabilities <= 0)[0]
                included_txs.append(zero_indices)
        else:
            raise ValueError

        unique_txs = set(included_txs)
        num_unique_txs = len(unique_txs)
        throughput_tps = num_unique_txs / self.delta
        optimal_throughput_tps = self.num_miners * self.b / self.delta
        total_private_value = sum(self.state[tx_id] for tx_id in unique_txs) # social welfare

        # update state
        done = True
        self.reset()

        return self.state, rewards, done, \
            {"probabilities": probabilities, "throughput": throughput_tps, \
             "optimal": optimal_throughput_tps, "total_private_value": total_private_value}

    def find_optim_action(self, actions, idx=0, cnt=100):
        assert idx >= 0 and idx < self.num_agents
        s = self.state[idx]
        max_reward = -1
        optim_action = 0
        action_copy = actions.copy()
        for a in np.linspace(0, s, cnt):
            action_copy[idx] = a
            rewards, _ = self.calculate_rewards(action_copy)
            # print(a, ',', rewards[idx])
            if rewards[idx] > max_reward:
                max_reward = rewards[idx]
                optim_action = a
        return optim_action, max_reward

    def calculate_probabilities(self, actions: np.ndarray) -> np.ndarray:
        # probability of being included in the block
        # if action <= 0, then probability = 0
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
            actions_burn = np.where(actions >= 0, self.a * np.log(1 + actions / self.a), actions)
            probabilities = self.calculate_probabilities(actions_burn)
        else:
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
    from tqdm import tqdm
    from easydict import EasyDict as edict
    from utils.utils import fix_seed

    BASE_CONFIGS_PATH = r'./config/base.yaml'

    with open(BASE_CONFIGS_PATH, 'r') as cfg_file:
        base_cfgs = yaml.load(cfg_file, Loader=yaml.FullLoader)
    base_cfgs = edict(base_cfgs)

    fix_seed(base_cfgs.seed)
    env = DAGEnv(
            fee_data_path=base_cfgs.fee_data_path,
            is_clip=base_cfgs.is_clip,
            clip_value=base_cfgs.clip_value,
            max_agents_num=base_cfgs.max_agents_num,
            lambd=base_cfgs.lambd,
            delta=base_cfgs.delta,
            b=base_cfgs.b,
            a=base_cfgs.a,
            is_burn=base_cfgs.is_burn
    )
    state = env.reset()

    # env.plot(env.f, (0, 1))
    # env.plot(env.invf, (env.f(1), 1))
    # env.plot(env.invf_with_clip, (0, 1))
    
    progress_bar = tqdm(range(1000))
    for _ in progress_bar:
        state, _, _ , info = env.step(state)
        progress_bar.set_description(f'throughout: {info["throughput"]}, optimal: {info["optimal"]}')
        print(f'throughout: {info["throughput"]}, optimal: {info["optimal"]}')
