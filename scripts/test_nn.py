import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.utils import fix_seed, log
from tqdm import tqdm
import csv
from agents.nn import Net


class Tester(object):
    def __init__(self, cfgs):
        super(Tester, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed + 666)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        
        self.agent = Net(num_agents=cfgs.max_agents_num, num_actions=cfgs.max_agents_num, lr=cfgs.train.lr).to(self.device)
        self.agent.load_state_dict(torch.load(cfgs.path.model_path))

    def testing(self):
        state = self.env.reset()
        throughput_list = []
        sw_list = []
        
        progress_bar = tqdm(range(1, self.cfgs.test.steps+1))
        for _ in progress_bar:
            actions = self.agent(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
            state, _, _, info = self.env.step(actions)
            throughput_list.append(info["throughput"])
            sw_list.append(info["total_private_value"])
            progress_bar.set_description(f'throughput: {throughput_list[-1]}, social welfare: {sw_list[-1]}')

        log(f'mean throughput: {np.mean(throughput_list)}, mean social welfare: {np.mean(sw_list)}')
