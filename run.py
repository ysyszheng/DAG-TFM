import gym
import torch
import numpy as np
from envs.AuctionEnv import AuctionEnv
from agents.ddpg import DDPG
from config.cfg import cfg
from utils.replay_buffer import ReplayBuffer
from utils.fix_seed import fix_seed
from tqdm import tqdm
from utils.log import log
import os

if __name__ == '__main__':
    env = gym.make('gym_auction_env-v0',
                   max_agents_num=cfg['max_agents_num'],
                   lambd=cfg['lambd'],
                   delta=cfg['delta'],
                   b=cfg['b'],
                   is_burn=cfg['is_burn'])
    fix_seed(cfg['seed'])
    replay_buffer = ReplayBuffer()
    agent = DDPG(
        1,
        1,
        cfg['lr'],
        cfg['gamma'],
        cfg['tau'],
        cfg['batch_size'],
    )
    rewards_lst = []

    dirs = ['./img', './models', './data']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    state = env.reset()
    # ! normalize here
    state = torch.tensor(state, dtype=torch.float32)

    progress_bar = tqdm(range(1, cfg['steps']+1))
    for step in progress_bar:
        cfg['num_miners'] = env.num_miners
        action = np.zeros_like(state)
        for i in range(len(state)):
            action[i] = agent.select_action_with_noise(
                state[i], cfg['noise_std'], cfg['epsilon_min'], cfg['epsilon_decay'])
        next_state, reward, done, info = env.step(action)

        for s, a, r, p, n in zip(state.data.numpy().flatten(),
                     action, reward, info['probabilities'], cfg['num_miners']*info['probabilities']):
            print(step, ',', s, ',', a, ',', a/s, ',', r, ',', p, ',', cfg['num_miners'], ',', n)

        next_state = torch.tensor(next_state, dtype=torch.float32)
        for i in range(len(state)):
            replay_buffer.add((state[i], action[i], reward[i]))
        state = next_state

        if len(replay_buffer) > cfg['batch_size']:
            agent.update(replay_buffer, cfg['iterations'])
        if done:
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)

        rewards_lst.append(sum(reward))
        progress_bar.set_description(f"total reward: {rewards_lst[-1]}")

        if step % cfg['save_freq'] == 0:
            torch.save(agent.actor.state_dict(), f'./models/actor_{step}.pth')
            torch.save(agent.critic.state_dict(),
                       f'./models/critic_{step}.pth')
            np.save(f'./data/rewards_{step}.npy', rewards_lst)
