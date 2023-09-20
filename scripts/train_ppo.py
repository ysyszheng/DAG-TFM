import gym
import numpy as np
import torch
from envs.DAGEnv import DAGEnv
from agents.ppo import PPO
from utils.utils import fix_seed, log, normize, unormize
from tqdm import tqdm
from easydict import EasyDict as edict
import csv


class Trainer(object):
    def __init__(self, cfgs: edict):
        super(Trainer, self).__init__()
        self.cfgs = cfgs
        fix_seed(cfgs.seed)

        self.env = gym.make('gym_dag_env-v0', 
            fee_data_path=cfgs.fee_data_path, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = PPO(1, 1, cfgs.model.actor_lr, cfgs.model.critic_lr, 
            cfgs.model.c1, cfgs.model.c2, cfgs.model.K_epochs, cfgs.model.gamma, 
            cfgs.model.eps_clip, cfgs.model.std_init, cfgs.model.std_decay, cfgs.model.std_min
        )

    def training(self):
        state = self.env.reset()
        state = normize(state, self.env.state_mean, self.env.state_std)
        reward_list = []
        sw_list = []

        with open(self.cfgs.path.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["update step", "private value", "tx fee", "incentive awareness", 
                             "revenue", "probability", "miner numbers"])

        progress_bar = tqdm(range(1, self.cfgs.train.steps+1))
        for step in progress_bar:
            action = np.zeros_like(state)

            for i in range(len(state)):
                action[i] = self.agent.select_action(state[i])

            state = unormize(state, self.env.state_mean, self.env.state_std)
            action = action * state

            next_state, reward, _, info = self.env.step(action)
            self.agent.buffer.rewards.extend(reward)

            with open(self.cfgs.path.log_path, 'a', newline='') as csvfile:
                for s, a, r, p in zip(state, action, reward, info['probabilities']):
                    writer = csv.writer(csvfile)
                    writer.writerow([int(step), s, a, a/s, r, p, self.env.num_miners])
            
            state = normize(next_state, self.env.state_mean, self.env.state_std)

            if step % self.cfgs.train.update_freq == 0:
                # * update models
                log('Update...')
                self.agent.update()
                
                # * test models
                action = np.zeros_like(state)

                for i in range(len(state)):
                    action[i] = self.agent.select_action_without_exploration(state[i]) # ? no exploration in testing, no grad

                state = unormize(state, self.env.state_mean, self.env.state_std)
                action = action * state

                next_state, reward, _, info = self.env.step(action)
                state = normize(next_state, self.env.state_mean, self.env.state_std)

                # * print network gradient
                log('Print network gradient...') # ! what is the gradient of the network?
                for name, param in self.agent.actor.named_parameters():
                    if param.requires_grad:
                        log(f'Actor: {name}, gradient: {param.grad}')
                for name, param in self.agent.critic.named_parameters():
                    if param.requires_grad:
                        log(f'Critic: {name}, gradient: {param.grad}')

                # * save rewards and social warfare (all data during training, not only test)
                reward_list.extend(reward)
                sw_list.append(sum(reward))
                
                np.save(self.cfgs.path.rewards_path, np.array(reward_list))
                np.save(self.cfgs.path.sw_path, np.array(sw_list))

                # * save models
                log('Save models...')
                torch.save(self.agent.actor.state_dict(), self.cfgs.path.actor_model_path)
                torch.save(self.agent.critic.state_dict(), self.cfgs.path.critic_model_path)

            if step % self.cfgs.train.std_decay_freq == 0:
                self.agent.decay_action_std()
                log(f'Decay action std to {self.agent.actor.std}')

            progress_bar.set_description(f"step: {step}, reward: {sum(reward)}")
