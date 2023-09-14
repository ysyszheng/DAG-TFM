import gym
import numpy as np
import torch
from envs.DAGEnv import DAGEnv
from agents.ppo import PPO
from utils.utils import fix_seed
from tqdm import tqdm
from utils.utils import log
from easydict import EasyDict as edict


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = PPO(1, 1, cfgs.actor_lr, cfgs.critic_lr, cfgs.c1, cfgs.c2, 
            cfgs.K_epochs, cfgs.gamma, cfgs.eps_clip, cfgs.std_init, cfgs.std_decay, cfgs.std_min
        )

    def training(self):
        state = self.env.reset()
        state_list = []
        action_list = []
        reward_list = []
        social_welfare_list = []

        progress_bar = tqdm(range(1, self.cfgs.steps+1))
        for step in progress_bar:
            action = np.zeros_like(state)

            for i in range(len(state)):
                action[i] = self.agent.select_action(state[i])

            next_state, reward, _, _ = self.env.step(action * state)
            self.agent.buffer.rewards.extend(reward)
            state = next_state

            if step % self.cfgs.test_freq == 0:
                action = np.zeros_like(state)

                for i in range(len(state)):
                    action[i] = self.agent.select_action_test(state[i])

                next_state, reward, _, _ = self.env.step(action * state)

                state_list.extend(state)
                action_list.extend(action)
                reward_list.extend(reward)
                social_welfare_list.append(sum(reward))
                
                state = next_state
                np.save(f"./state_history.npy", np.array(state_list))
                np.save(f"./action_history.npy", np.array(action_list))
                np.save(f"./rewards_history.npy", np.array(reward_list))
                np.save(f"./social_welfare_history.npy", np.array(social_welfare_list))

            if step % self.cfgs.update_freq == 0:
                log('Update...')
                self.agent.update()

            if step % self.cfgs.std_decay_freq == 0:
                self.agent.decay_action_std()
                log(f'Decay action std to {self.agent.actor.std}')

            if step % self.cfgs.save_freq == 0:
                log('Save...')
                torch.save(self.agent.actor.state_dict(), self.cfgs.actor_path)
                torch.save(self.agent.critic.state_dict(), self.cfgs.critic_path)

            progress_bar.set_description(f"step: {step}, reward: {sum(reward)}")
