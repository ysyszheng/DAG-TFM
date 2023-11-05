import numpy as np
from envs.DAGEnv import DAGEnv
from agents.es import Net
from utils.utils import log, fix_seed
from tqdm import tqdm
from easydict import EasyDict as edict
import csv
import torch
import gym


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, 
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )

        self.strategies = Net(num_agents=1, num_actions=1).to(self.device)
        self.new_strategies = Net(num_agents=1, num_actions=1).to(self.device)
        
        total_params = sum(p.numel() for p in self.strategies.parameters())
        self.J = int(4 + 3 * np.floor(np.log(total_params))) # 25


    def OriginalMinusRegret(self, alpha1=.01, mu1=.1):
        '''consider regret of agent 0
        original algorithm
        '''
        self.new_strategies.load_state_dict(self.strategies.state_dict())
        state = self.env.reset()

        # Oracle
        V = 0
        for _ in range(self.cfgs.train.oracle_query_times):
            action = self.strategies(torch.FloatTensor(state).to(torch.float64)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            state, reward, _ = self.env.step_without_packing(action)
            V += reward[0]
        V /= self.cfgs.train.oracle_query_times

        # NES
        for _ in tqdm(range(self.cfgs.train.iterations)):
            epsilon = np.random.normal(0, 1, size=self.J)
            r = np.zeros(self.J)

            for j in range(self.J):
                for param in self.new_strategies.parameters():
                    param.data += mu1 * epsilon[j]
                
                # Oracle
                r_j_plus = 0
                for _ in range(self.cfgs.train.oracle_query_times):
                    action0 = self.new_strategies(torch.FloatTensor(state[0].reshape(-1, 1)).to(torch.float64).to(self.device))\
                      .reshape(1).detach().cpu().numpy()
                    action = self.strategies(torch.FloatTensor(state[1:]).reshape(-1, 1).to(torch.float64).to(self.device))\
                      .squeeze().detach().cpu().numpy()
                    action = np.concatenate((action0, action))
                    state, reward, _ = self.env.step_without_packing(action)
                    r_j_plus += reward[0]
                r_j_plus /= self.cfgs.train.oracle_query_times

                for param in self.new_strategies.parameters():
                    param.data -= 2 * mu1 * epsilon[j]

                # Oracle
                r_j_minus = 0
                for _ in range(self.cfgs.train.oracle_query_times):
                    action0 = self.new_strategies(torch.FloatTensor(state[0].reshape(-1, 1)).to(torch.float64).to(self.device))\
                      .reshape(1).detach().cpu().numpy()
                    action = self.strategies(torch.FloatTensor(state[1:]).reshape(-1, 1).to(torch.float64).to(self.device))\
                      .squeeze().detach().cpu().numpy()
                    action = np.concatenate((action0, action))
                    state, reward, _ = self.env.step_without_packing(action)
                    r_j_minus += reward[0]
                r_j_minus /= self.cfgs.train.oracle_query_times

                r[j] = (r_j_plus - r_j_minus)

                for param in self.new_strategies.parameters():
                    param.data += mu1 * epsilon[j]
            
            for param in self.new_strategies.parameters():
                param.data += alpha1 / (self.J * mu1) * sum(r * epsilon)

        # Oracle
        DEV = 0
        for _ in range(self.cfgs.train.oracle_query_times):
            action0 = self.new_strategies(torch.FloatTensor(state[0].reshape(-1, 1)).to(torch.float64).to(self.device))\
              .reshape(1).detach().cpu().numpy()
            action = self.strategies(torch.FloatTensor(state[1:]).reshape(-1, 1).to(torch.float64).to(self.device))\
              .squeeze().detach().cpu().numpy()
            action = np.concatenate((action0, action))
            state, reward, _ = self.env.step_without_packing(action)
            DEV += reward[0]
        DEV /= self.cfgs.train.oracle_query_times

        log(f'V: {V}, DEV: {DEV}')
        return V - DEV


    def MinusRegret(self):
        """consider regret of agent 0
        directly use `env.find_optim_action()`
        """
        state = self.env.reset()

        regret = 0
        progress_bar = tqdm(range(self.cfgs.train.oracle_query_times))
        for r in progress_bar:
            action = self.strategies(torch.FloatTensor(state).to(torch.float64)\
              .reshape(-1, 1).to(self.device)).squeeze().detach().cpu().numpy()
            _, opt_reward = self.env.find_optim_action(action, idx=0)
            state, reward, _ = self.env.step_without_packing(action)
            regret += (opt_reward - reward[0]) / state[0]
            progress_bar.set_description(f'regret: {regret / (r + 1)}')

        regret /= self.cfgs.train.oracle_query_times
        return regret

    
    def MiniMax(self, alpha1, mu1, alpha2, mu2):
        for iter in range(self.cfgs.train.iterations):
            print(f'************* iter: {iter} *************')
            epsilon = np.random.normal(0, 1, size=self.J)
            r = np.zeros(self.J)

            # NES
            for j in range(self.J):
                for param in self.strategies.parameters():
                    param.data += mu2 * epsilon[j]

                # r_j_plus = self.OriginalMinusRegret(alpha1, mu1)
                r_j_plus = self.MinusRegret()

                for param in self.strategies.parameters():
                    param.data -= 2 * mu2 * epsilon[j]

                # r_j_minus = self.OriginalMinusRegret(alpha1, mu1)
                r_j_minus = self.MinusRegret()

                r[j] = (r_j_plus - r_j_minus)

                for param in self.strategies.parameters():
                    param.data += mu2 * epsilon[j]

            for param in self.strategies.parameters():
                param.data += alpha2 / (self.J * mu2) * sum(r * epsilon)

            torch.save(self.strategies.state_dict(), self.cfgs.path.model_path)

        # return self.MinusRegret()
                

    def training(self):
        self.MiniMax(self.cfgs.train.alpha1, self.cfgs.train.mu1, self.cfgs.train.alpha2, self.cfgs.train.mu2)
        torch.save(self.strategies.state_dict(), self.cfgs.path.model_path)
