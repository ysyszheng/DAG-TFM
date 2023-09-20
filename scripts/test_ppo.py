import gym
import numpy as np
import torch
from envs.DAGEnv import DAGEnv
from agents.ppo import PPO
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
from easydict import EasyDict as edict


class Tester(object):
    def __init__(self, cfgs: edict):
        super(Tester, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed+666) # * use different seed from training

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = PPO(1, 1, cfgs.actor_lr, cfgs.critic_lr, cfgs.c1, cfgs.c2, 
            cfgs.K_epochs, cfgs.gamma, cfgs.eps_clip, cfgs.std_init, cfgs.std_decay, cfgs.std_min
        )

        self.agent.actor.load_state_dict(torch.load(cfgs.actor_path))
        self.agent.critic.load_state_dict(torch.load(cfgs.critic_path))

    def testing(self):
        state = self.env.reset()
        state = normize(state, self.env.state_mean, self.env.state_std)
        throughput_list = []
        total_private_value_list = []

        progress_bar = tqdm(range(1, self.cfgs.test.steps+1))
        for _ in progress_bar:
            action = np.zeros_like(state)

            for i in range(len(state)):
                action[i] = self.agent.select_action_without_exploration(state[i])

            state = unormize(state, self.env.state_mean, self.env.state_std)
            action = action * state

            next_state, _, _, info = self.env.step(action)

            state = normize(next_state, self.env.state_mean, self.env.state_std)

            throughput_list.append(info['throughput'])
            total_private_value_list.append(info['total_private_value'])

            progress_bar.set_description(f'Throughput: {info["throughput"]:.2f} | Total Private Value: {info["total_private_value"]:.2f}')

        log(f'Throughput: {np.mean(throughput_list):.2f} | Total Private Value: {np.mean(total_private_value_list):.2f}')
