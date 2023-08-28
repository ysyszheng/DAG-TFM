import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--window_size', type=int, default=100)
    args = parser.parse_args()
    
    steps = args.steps
    window_size = args.window_size
    rewards = np.load(f'./rewards/rewards_{steps}.npy')
    moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(18, 6))
    plt.plot(rewards, label='Rewards')
    plt.plot(moving_average, label=f'Moving Average ({window_size})')
    plt.legend()
    plt.savefig(f'./img/rewards_{steps}.png')
    plt.show()
