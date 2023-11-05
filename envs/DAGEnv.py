from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import gym
import concurrent.futures as cf
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve, root_scalar, minimize_scalar
from gym import spaces
import numpy as np
import random
from utils.utils import log, fix_seed, get_dist_from_margin, print_thread_count
import time
from copy import deepcopy


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
        self.fee_list = self.fee_list[self.fee_list > 0] # remove 0
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
        # miner numbers in delta time, Poisson distribution or fixed number
        self.num_miners = 1 + np.random.poisson(self.lambd * self.delta)
        # self.num_miners = self.lambd * self.delta

        return self.state
    

    def step_without_packing(self, actions):
        # calculate rewards and probabilities
        rewards, _ = self.calculate_rewards(actions)
        # update state
        done = True
        self.reset()
        return self.state, rewards, done


    def step(self, actions):
        # calculate rewards and probabilities
        rewards, probabilities = self.calculate_rewards(actions)

        # calculate throughput and social welfare (total private value)
        included_txs = []
        # use marginal probabilities
        for _ in range(self.num_agents):
            random_numbers = np.random.rand(self.num_agents)
            mask = random_numbers < probabilities
            selected_indices = np.where(mask)[0].tolist()
            included_txs.extend(selected_indices)

        unique_txs = set(included_txs)
        included_txs_bool = np.zeros_like(self.state)
        for i in unique_txs:
            included_txs_bool[i] = 1
        num_unique_txs = len(unique_txs)
        throughput_tps = num_unique_txs / self.delta
        optimal_throughput_tps = self.num_miners * self.b / self.delta
        total_private_value = sum(self.state[tx_id] for tx_id in unique_txs) # social welfare

        # update state
        done = True
        self.reset()

        return self.state, rewards, done, \
            {"probabilities": probabilities, "throughput": throughput_tps, \
             "optimal": optimal_throughput_tps, "total_private_value": total_private_value, \
             "included_txs": included_txs_bool}


    def find_optim_action(self, actions, idx=0):
        assert idx >= 0 and idx < self.num_agents
        def calculate_neg_rewards(a, *args):
            actions, idx = args
            actions_copy = deepcopy(actions)
            actions_copy[idx] = a
            rewards, _ = self.calculate_rewards(actions_copy)
            return -rewards[idx]

        s = self.state[idx]
        result = minimize_scalar(calculate_neg_rewards, args=(actions, idx), bounds=(0, s), method='bounded')
        optim_action = result.x
        max_reward = -result.fun
        # log(f'idx: {idx}, state: {self.state[idx]}, optim_action: {optim_action}, max_reward: {max_reward}')
        return optim_action, max_reward


    def find_all_optim_action(self, actions):
        optim_actions = np.zeros_like(actions)
        max_rewards = np.zeros_like(actions)

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(self.find_optim_action, 
                            [(actions, idx) for idx in range(self.num_agents)])
            for idx, result in enumerate(results):
                optim_actions[idx], max_rewards[idx] = result

        return optim_actions, max_rewards


    def find_nash_equilibrium(self, epsilon=1e-4): # assume has common knowledge
        action = np.random.random(self.state.shape) * self.state
        reward = self.calculate_rewards(action)[0]
        flag = False
        while not flag:
            optim_action, optim_reward = self.find_all_optim_action(action)
            loss = np.max((optim_reward-reward)/(reward+1e-6))
            if loss <= epsilon:
                flag = True
            action = optim_action
            reward = optim_reward
            # log(action, loss)
        return action, reward


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
            # with mp.Pool(processes=mp.cpu_count()) as pool:
            #     results = pool.map(self.invf_with_clip, z/(v[:l+1]))
            #     for h, result in enumerate(results):
            #         res += k[h] * result
            for h in range(l+1):
                res += k[h] * self.invf_with_clip(z/(v[h]))
            return (res - self.b)

        def GG(l): return G(int(l), v[int(l)]) # consider l as an integer

        if GG(0) > 0:
            raise ValueError # G(0,v[0]) = -b < 0
        elif GG(len(v)-1) <= 0:
            k_max = len(v)-1
        else:
            k_max = int(brentq(GG, 0, len(v)-1))
            if GG(k_max) > 0:
                k_max -= 1

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
    import cProfile

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
    
    # print('step', ',', 'private value', ',', 'optimal action')
    # for i in tqdm(range(100)):
    #     action, reward = env.find_nash_equilibrium()
        
    #     for s, a, r in zip(state, action, reward):
    #         print(i, ',', s,',',a ,',',r)


    # # just for test
    # action = state

    # start_time = time.time()
    # env.find_optim_action(action)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f'exec time: {execution_time} second') # 68s

    # start_time = time.time()
    # env.step(action)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f'exec time: {execution_time} second') # 1.6s
    
    # _, _, _ , info = env.step(state)
    # for fee, prob in zip(state, info['probabilities']):
    #     print(fee, ',', prob)


    # cProfile.run('env.step(state)', sort='cumtime')

    # env.plot(env.f, (0, 1))
    # env.plot(env.invf, (env.f(1), 1))
    # env.plot(env.invf_with_clip, (0, 1))

    # # plot
    # for lambd in range(1,10):
    #     env = DAGEnv(
    #             fee_data_path=base_cfgs.fee_data_path,
    #             is_clip=base_cfgs.is_clip,
    #             clip_value=base_cfgs.clip_value,
    #             max_agents_num=base_cfgs.max_agents_num,
    #             lambd=lambd,
    #             delta=base_cfgs.delta,
    #             b=base_cfgs.b,
    #             a=base_cfgs.a,
    #             is_burn=base_cfgs.is_burn
    #     )
    #     state = env.reset()

        
    #     throughput_list = []
    #     sw_list = []
    #     progress_bar = tqdm(range(3))
    #     for _ in progress_bar:
    #         state, _, _ , info = env.step(state)
    #         throughput_list.append(info["throughput"])
    #         sw_list.append(info['total_private_value'])
    #         progress_bar.set_description(f'throughout: {info["throughput"]}, optimal: {info["optimal"]}')
    #     print(f'lambda: {lambd}, throughout: {sum(throughput_list)/len(throughput_list)}, sw: {sum(sw_list)/len(sw_list)}')
