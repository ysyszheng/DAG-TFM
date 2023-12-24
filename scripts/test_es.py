import gym
import numpy as np
import torch
from envs.DAGEnv import DAGEnv
from agents.es import Net
from utils.utils import fix_seed, log, init_logger
from tqdm import tqdm
from easydict import EasyDict as edict


class Tester(object):
    def __init__(self, cfgs: edict):
        super(Tester, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)
        self.device = torch.device('cpu')

        self.strategies = Net(num_agents=1, num_actions=1).to(self.device)
        self.strategies.load_state_dict(torch.load(cfgs.path.model_path))

        self.env: DAGEnv = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.path.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, burn_flag=cfgs.burn_flag,
            clip_value=cfgs.clip_value, norm_value=cfgs.norm_value, log_file_path=cfgs.path.log_path,
        )

        self.logger = init_logger(__name__, self.cfgs.path.log_path)

    def testing(self):
        self.logger.info(f'========== {self.cfgs.method} Tester start, lambda: {self.cfgs.lambd}, burn flag: {self.cfgs.burn_flag}, a: {self.cfgs.a} ==========')

        state = self.env.reset()
        throughput_list = []
        sw_list = []
        
        for _ in range(self.cfgs.test.test_round):
            action = self.strategies(torch.FloatTensor(state).to(torch.float64)\
                .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            state, _, _ , info = self.env.step(action)
            throughput_list.append(info["throughput"])
            sw_list.append(info['social_welfare'])
        self.logger.info(f'throughout: {sum(throughput_list)/len(throughput_list)}, social welfare: {sum(sw_list)/len(sw_list)}')
        self.logger.info(f'========== {self.cfgs.method} Tester end, lambda: {self.cfgs.lambd}, burn flag: {self.cfgs.burn_flag}, a: {self.cfgs.a} ==========\n')
