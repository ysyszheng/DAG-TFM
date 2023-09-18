import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.utils import fix_seed
from tqdm import tqdm
from utils.utils import log
import csv
from agents.ppo import PPO


class Evaluator(object):
    def __init__(self, cfgs):
        super(Evaluator, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed + 666)

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = PPO(1, 1, cfgs.actor_lr, cfgs.critic_lr, cfgs.c1, cfgs.c2, 
            cfgs.K_epochs, cfgs.gamma, cfgs.eps_clip, cfgs.std_init, cfgs.std_decay, cfgs.std_min
        )
        self.agent.actor.load_state_dict(torch.load(cfgs.actor_model_path))
        self.agent.critic.load_state_dict(torch.load(cfgs.critic_model_path))

    def evaluating(self):
        pass