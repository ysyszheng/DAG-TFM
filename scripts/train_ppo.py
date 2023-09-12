import gym
import numpy as np
from envs.DAGEnv import DAGEnv
from agents.ppo import PPO
from utils.fix_seed import fix_seed
from tqdm import tqdm
from utils.log import log
from easydict import EasyDict as edict


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)

        self.env = gym.make('gym_dag_env-v0', max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = PPO(1, 1, cfgs.actor_lr, cfgs.critic_lr, cfgs.c1, 
            cfgs.c2, cfgs.K_epochs, cfgs.gamma, cfgs.eps_clip,
        )

    def training(self):
        state = self.env.reset()
        reward_list = []

        progress_bar = tqdm(range(1, self.cfgs.steps+1))
        for step in progress_bar:
            action = np.zeros_like(state)

            for i in range(len(state)):
                action[i] = self.agent.select_action(state[i])

            next_state, reward, _, _ = self.env.step(action * state)
            reward_list.append(sum(reward))
            self.agent.buffer.rewards.extend(reward)
            state = next_state

            if step % self.cfgs.update_freq == 0:
                log('Update...')
                self.agent.update()

            if step % self.cfgs.save_freq == 0:
                log('Save...')
                np.save(f"./rewards_history.npy", np.array(reward_list))

            progress_bar.set_description(f"step: {step}, reward: {sum(reward)}")
