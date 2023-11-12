import gym
import numpy as np
import torch
from envs.DAGEnv import DAGEnv
from agents.es import Net
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
from easydict import EasyDict as edict


class Tester(object):
    def __init__(self, cfgs: edict):
        super(Tester, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed+666) # * use different seed from training
        self.device = torch.device('cpu')

        self.strategies = Net(num_agents=1, num_actions=1).to(self.device)
        self.strategies.load_state_dict(torch.load(cfgs.path.model_path))

    def testing(self):
        for lambd in range(1,self.cfgs.test.max_lambd+1):
            env = DAGEnv(
                    fee_data_path=self.cfgs.fee_data_path,
                    is_clip=self.cfgs.is_clip,
                    clip_value=self.cfgs.clip_value,
                    max_agents_num=self.cfgs.max_agents_num,
                    lambd=lambd,
                    delta=self.cfgs.delta,
                    b=self.cfgs.b,
                    a=self.cfgs.a,
                    is_burn=self.cfgs.is_burn
            )
            state = env.reset()
            
            throughput_list = []
            sw_list = []
            
            for _ in range(self.cfgs.test.test_round):
                action = self.strategies(torch.FloatTensor(state).to(torch.float64)\
                    .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
                state, _, _ , info = env.step(action)
                throughput_list.append(info["throughput"])
                sw_list.append(info['total_private_value'])
            print(f'lambda: {lambd}, throughout: {sum(throughput_list)/len(throughput_list)}, sw: {sum(sw_list)/len(sw_list)}')
