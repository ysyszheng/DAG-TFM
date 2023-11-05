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
            fee_data_path=cfgs.fee_data_path, is_clip=cfgs.is_clip,
            clip_value=cfgs.clip_value, max_agents_num=cfgs.max_agents_num,
            lambd=cfgs.lambd, delta=cfgs.delta, a=cfgs.a, b=cfgs.b, is_burn=cfgs.is_burn,
        )
        self.agent = PPO(1, 1, cfgs.model.actor_lr, cfgs.model.critic_lr, 
            cfgs.model.c1, cfgs.model.c2, cfgs.model.K_epochs, cfgs.model.gamma, 
            cfgs.model.eps_clip, cfgs.model.std_init, cfgs.model.std_decay, cfgs.model.std_min
        )

    def training(self):
        state = self.env.reset()
        reward_list = []
        sw_list = []
        epsilon_list = []

        with open(self.cfgs.path.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["step", "private value", "tx fee", "optimal tx fee", 
                        "revenue", "optimal revenue", "epsilon", "reward shaped",
                        "probability", "incentive awareness", "miner numbers", "is included"])

        progress_bar = tqdm(range(1, self.cfgs.train.steps+1))
        for step in progress_bar:
            action = np.zeros_like(state)
            for i in range(len(state)):
                action[i] = self.agent.select_action(state[i])
            
            optimal_action, optimal_reward = self.env.find_all_optim_action(action)

            next_state, reward, _, info = self.env.step(action)

            optimal_reward = np.where(optimal_reward >= reward, optimal_reward, reward)
            optimal_action = np.where(optimal_reward >= reward, optimal_action, action)

            epsilon = (optimal_reward - reward)/state # epsilon-bayesian-nash
            r_shaped = -epsilon * 1e3
            self.agent.buffer.rewards.extend(r_shaped)

            reward_list.append(reward)
            sw_list.append(info["total_private_value"])
            epsilon_list.append(epsilon)

            with open(self.cfgs.path.log_path, 'a', newline='') as csvfile:
                for s, a, oa, rev, orev, e, rs, p, included in zip(state, action, 
                                    optimal_action, reward, optimal_reward, epsilon, 
                                    r_shaped, info['probabilities'], info['included_txs']):
                    writer = csv.writer(csvfile)
                    writer.writerow([int(step), s, a, oa, rev, orev, e, rs, p, a/s, self.env.num_miners, int(included)])

            state = next_state

            progress_bar.set_description(f"step: {step}, epsilon: {np.max(epsilon):.4f}, r_shaped: {np.min(r_shaped):.4f}, std: {self.agent.actor.std:.4f}")
            print(f'step: {step}, epsilon: {np.max(epsilon):.4f}, r_shaped: {np.min(r_shaped):.4f}, std: {self.agent.actor.std:.4f}')

            if step % self.cfgs.train.update_freq == 0:
                # * update models
                log('Update models & Save datas & Save models...')
                self.agent.update()
                
                # * save datas
                np.save(self.cfgs.path.rewards_path, np.array(reward_list))
                np.save(self.cfgs.path.sw_path, np.array(sw_list))
                np.save(self.cfgs.path.epsilon_path, np.array(epsilon_list))

                # * save models
                torch.save(self.agent.actor.state_dict(), self.cfgs.path.actor_model_path)
                torch.save(self.agent.critic.state_dict(), self.cfgs.path.critic_model_path)

            if step % self.cfgs.train.std_decay_freq == 0:
                self.agent.decay_action_std()
                log(f'Decay action std to {self.agent.actor.std}')

