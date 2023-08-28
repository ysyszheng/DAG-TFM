import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.fix_seed import fix_seed
from tqdm import tqdm
from utils.log import log
import csv


class Linear(object):
    def __init__(self, cfg):
        super(Linear, self).__init__()
        self.cfg = cfg
        fix_seed(cfg['seed'])

        self.env = gym.make(
            'gym_dag_env-v0',
            max_agents_num=cfg['max_agents_num'],
            lambd=cfg['lambd'],
            delta=cfg['delta'],
            b=cfg['b'],
            is_burn=cfg['is_burn']
        )
    
    def optim(self):
        state = self.env.reset()
        # progress_bar = tqdm(range(1, 10+1))
        for step in range(1, 10+1):
            max_rev, ia = 0, 0
            progress_bar = tqdm(np.linspace(0, 1, 100))
            for k in progress_bar:
                action = state * k
                rewards, _ = self.env.calculate_rewards(action)
                if rewards[0] > max_rev:
                    max_rev, ia = rewards[0], k
                    progress_bar.set_description(f"incentive awareness: {ia}, social warfare: {sw}")
            # progress_bar.set_description(f"incentive awareness: {ia}, social warfare: {sw}")
            print(f"step: {step}, incentive awareness: {ia}, social warfare: {sw}")

            state, _, done, _ = self.env.step(state * ia)

            if done:
                state = self.env.reset()
