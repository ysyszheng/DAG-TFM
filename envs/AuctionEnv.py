from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
from pynverse import inversefunc
# from utils.get_fee import get_fees_from_files
from gym import spaces
import numpy as np
import random
np.seterr(divide='ignore', invalid='ignore')
import warnings
import powerlaw
import pickle
warnings.filterwarnings("ignore")


class AuctionEnv(gym.Env):
    def __init__(self,
                 max_agents_num,
                 lambd,
                 delta,
                 b,
                 is_burn=False):
        self.max_agents_num = max_agents_num
        self.lambd = lambd
        self.delta = delta
        self.b = b
        # self.num_agents = np.random.randint(self.max_agents_num) # TODO:
        self.num_agents = self.max_agents_num
        self.is_burn = is_burn
        with open('./fee.pkl', 'rb') as file:
            self.fee_list = pickle.load(file)
        self.max_private_value = max(self.fee_list)

        # action: transaction fee
        self.action_space = spaces.Box(
            low=0, high=np.inf, shape=(max_agents_num,), dtype=np.float32)
        # state: private values of agents and number of agents (padding 0)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(max_agents_num,), dtype=np.float32)

        self.f = lambda p: np.where(p == 0, 1, np.divide(
            (1 - np.exp(-self.lambd * self.delta * p)), (self.lambd * self.delta * p), dtype=np.float64)) # FIXME: divide by zero
        self.invf = inversefunc(self.f)

    def reset(self):
        self.num_agents = np.random.randint(self.max_agents_num)
        self.num_agents = self.max_agents_num
        self.state = np.array(random.sample(self.fee_list, self.num_agents))
        self.num_miners = 1 + np.random.poisson(self.lambd)

        return self.state

    def step(self, actions):
        rewards, probabilities = self.calculate_rewards(actions)
        done = True
        self.reset()

        return self.state, rewards, done, {"probabilities": probabilities}

    def calculate_probabilities(self, actions: np.ndarray) -> np.ndarray:
        probabilities = np.zeros(self.num_agents)
        v, k = np.unique(actions, return_counts=True)
        v = v[::-1]

        def G(l, z):
            assert l >= 0 and l < self.num_agents
            res = 0
            for h in range(l+1):
                res += k[h] * np.minimum(self.invf(z/v[h]), 1)
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
        # c_k_max = brentq(G_k_max, v[-1], v[k_max])
        c_k_max = fsolve(G_k_max, x0=v[k_max])[0]

        p = [0 for _ in range(self.num_agents)]
        for l in range(k_max+1):  # k[l] != 0
            p[l] = min(self.invf(c_k_max/v[l]), 1)

        for i in range(self.num_agents):
            print(v, actions, np.where(actions[i] == v))
            idx = np.where(actions[i] == v)[0][0]
            probabilities[i] = p[idx]
        return probabilities

    def calculate_rewards(self, actions):
        rewards = np.zeros(self.num_agents)
        # ! burn here
        if self.is_burn:
            probabilities = self.calculate_probabilities(np.log(1+actions))
            # probabilities = self.calculate_probabilities(np.sqrt(actions))
        else:
            probabilities = self.calculate_probabilities(actions)

        for i in range(self.num_agents):
            private_value = self.state[i]
            offer_price = actions[i]
            probability = probabilities[i]

            # ?
            if offer_price < private_value:
                rewards[i] =  (1 - (1 - probability) ** self.num_miners) * \
                    (private_value - offer_price)
            else:
                rewards[i] = 0
            # rewards[i] =  (1 - (1 - probability) ** self.num_miners) * \
            #         (private_value - offer_price)

        return rewards, probabilities

    def plot(self, func=None, bounds=(-5, 5)):
        x = np.linspace(bounds[0], bounds[1], 1000)
        if func is not None:
            y = func(x)
        plt.plot(x, y)
        plt.title(func.__name__)
        plt.show()
