import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def plot_rewards(file_path, window_size):
    rewards = np.load(file_path)
    moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.plot(rewards, label='Rewards', alpha=0.5, color='blue')
    plt.plot(moving_average, label=f'Moving Average ({window_size})', color='red')
    plt.legend()
    plt.savefig(f'./results/img/{os.path.splitext(os.path.basename(file_path))[0]}.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default=None, help='File Path')
    parser.add_argument('--window_size', type=int, default=100, help='Window Size')
    args = parser.parse_args()

    plot_rewards(args.file_path, args.window_size)
