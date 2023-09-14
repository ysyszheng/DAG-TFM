import gym
import numpy as np
import torch
from envs.DAGEnv import DAGEnv
from agents.ppo import PPO
from utils.utils import fix_seed
from tqdm import tqdm
from utils.utils import log
from easydict import EasyDict as edict


class Tester(object):
    def __init__(self, cfgs: edict):
        super(Tester, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed+666)

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = PPO(1, 1, cfgs.actor_lr, cfgs.critic_lr, cfgs.c1, cfgs.c2, 
            cfgs.K_epochs, cfgs.gamma, cfgs.eps_clip, cfgs.std_init, cfgs.std_decay, cfgs.std_min
        )

        self.agent.actor.load_state_dict(torch.load(cfgs.actor_path))
        self.agent.critic.load_state_dict(torch.load(cfgs.critic_path))

    def testing(self):
        state = self.env.reset()
        