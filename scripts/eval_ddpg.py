import gym
import torch
import numpy as np
from envs.DAGEnv import DAGEnv
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
import csv
from agents.ddpg import DDPG


class Evaluator(object):
    def __init__(self, cfgs):
        super(Evaluator, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed + 666)

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = DDPG(1, 1, cfgs.model.actor_lr, cfgs.model.critic_lr, cfgs.model.gamma, cfgs.model.tau,
            cfgs.model.std, cfgs.model.batch_size, cfgs.model.epsilon_min, cfgs.model.epsilon_decay
        )
        self.agent.actor.load_state_dict(torch.load(cfgs.path.actor_model_path))
        self.agent.critic.load_state_dict(torch.load(cfgs.path.critic_model_path))

    def evaluating(self):
        state = self.env.reset()
        state = normize(state, self.env.state_mean, self.env.state_std)
        specific_action_list = []
        optimal_action_list = []
        specific_reward_list = []
        optimal_reward_list = []
        
        progress_bar = tqdm(range(1, self.cfgs.eval.steps+1))
        for _ in progress_bar:
            action = np.zeros_like(state)

            for i in range(len(state)):
                action[i] = self.agent.select_action(state[i])

            state = unormize(state, self.env.state_mean, self.env.state_std)
            action = action * state
            specific_action = action[self.cfgs.eval.idx]

            optimal_action, max_reward = self.env.find_optim_action(actions=action, 
                    idx=self.cfgs.eval.idx, cnt=self.cfgs.eval.cnt)

            specific_action_list.append(specific_action)
            optimal_action_list.append(optimal_action)
            
            next_state, reward, _, _ = self.env.step(action)
            state = normize(next_state, self.env.state_mean, self.env.state_std)

            specific_reward = reward[self.cfgs.eval.idx]
            specific_reward_list.append(specific_reward)
            optimal_reward_list.append(max_reward)

            progress_bar.set_description(f'a_opt: {optimal_action:.3f}, a: {specific_action:.3f}, r_opt: {max_reward:.3f}, r:{specific_reward:.3f}, Deviation: {np.abs(specific_reward - max_reward) / max_reward:.5f}')

        relative_deviation = np.abs(np.array(specific_reward_list) - \
                np.array(optimal_reward_list)) / np.array(optimal_reward_list)
        log(f'Relative Deviation: {relative_deviation}')
