import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
import csv
from agents.ppo import PPO


class Evaluator(object):
    def __init__(self, cfgs):
        super(Evaluator, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed + 666)

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = PPO(1, 1, cfgs.actor_lr, cfgs.critic_lr, cfgs.c1, cfgs.c2, 
            cfgs.K_epochs, cfgs.gamma, cfgs.eps_clip, cfgs.std_init, cfgs.std_decay, cfgs.std_min
        )
        self.agent.actor.load_state_dict(torch.load(cfgs.actor_model_path))
        self.agent.critic.load_state_dict(torch.load(cfgs.critic_model_path))

    def evaluating(self):
        state = self.env.reset()
        state = normize(state, self.env.state_mean, self.env.state_std)
        specific_action_list = []
        optimal_action_list = []
        
        progress_bar = tqdm(range(1, self.cfgs.evaluation.steps+1))
        for _ in progress_bar:
            action = np.zeros_like(state)

            for i in range(len(state)):
                action[i] = self.agent.select_action_without_exploration(state[i])

            state = unormize(state, self.env.state_mean, self.env.state_std)
            action = action * state

            optimal_action = self.env.find_optim_action(action=action, 
                    idx=self.cfgs.eval.idx, cnt=self.cfgs.eval.cnt)

            specific_action = action[self.cfgs.eval.idx]
            specific_action_list.append(specific_action)
            optimal_action_list.append(optimal_action)
            
            next_state, _, _, _ = self.env.step(optimal_action)
            state = normize(next_state, self.env.state_mean, self.env.state_std)

        relative_deviation = np.abs(np.array(specific_action_list) - \
                np.array(optimal_action_list)) / np.array(optimal_action_list)
        log(f'Relative Deviation: {relative_deviation}')
