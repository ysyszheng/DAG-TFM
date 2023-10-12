import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.utils import fix_seed, log
from tqdm import tqdm
import csv
from agents.nn import Net


class Evaluator(object):
    def __init__(self, cfgs):
        super(Evaluator, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed + 666)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip, clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        
        self.agent = Net(num_agents=cfgs.max_agents_num, num_actions=cfgs.max_agents_num, lr=cfgs.train.lr).to(self.device)
        self.agent.load_state_dict(torch.load(cfgs.path.model_path))

    def evaluating(self):
        state = self.env.reset()
        delta_action_list = []
        delta_reward_list = []
        
        with open(self.cfgs.path.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Round", "private value", "tx fee", "revenue", 
                             "optimal tx fee", "optimal revenue", 
                             "incentive awareness", "probability",  "miner numbers"])

        progress_bar = tqdm(range(1, self.cfgs.eval.steps+1))
        for round in progress_bar:
            actions = self.agent(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
            optimal_rewards = np.zeros(self.cfgs.max_agents_num)
            optimal_actions = np.zeros(self.cfgs.max_agents_num)

            for idx in range(self.cfgs.max_agents_num):
                optimal_action, optimal_reward = self.env.find_optim_action(
                    actions=actions, idx=idx, cnt=self.cfgs.eval.cnt)
                optimal_actions[idx] = optimal_action
                optimal_rewards[idx] = optimal_reward

            next_state, rewards, _, info = self.env.step(actions)

            optimal_actions = np.where(optimal_rewards >= rewards, optimal_actions, actions)
            optimal_rewards = np.where(optimal_rewards >= rewards, optimal_rewards, rewards)

            with open(self.cfgs.path.log_path, 'a', newline='') as csvfile:
                for s, a, r, oa, or_, p in zip(state, actions, rewards, optimal_actions, optimal_rewards, info['probabilities']):
                    writer = csv.writer(csvfile)
                    writer.writerow([int(round), s, a, r, oa, or_, a/s, p, self.env.num_miners])

            state = next_state

            # delta_action_list.append(np.max(np.abs(optimal_actions - actions)/optimal_actions))
            # delta_reward_list.append(np.max((optimal_rewards - rewards)/optimal_rewards))
            delta_action_list.append(np.max(np.abs(optimal_actions - actions)/actions))
            delta_reward_list.append(np.max((optimal_rewards - rewards)/rewards))
            
            progress_bar.set_description(f'Max action deviation: {delta_action_list[-1]}, Max reward deviation: {delta_reward_list[-1]}')
        
        log(f'mean max action deviation: {np.mean(delta_action_list)}, mean max reward deviation: {np.mean(delta_reward_list)}')
