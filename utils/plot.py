import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot(file_path, label):
    data = np.load(file_path)
    # data = np.max(data, axis=1)
    
    plt.figure()
    plt.plot(data, label=label)
    plt.legend()
    plt.xlabel('Update Step')
    plt.ylabel(label)
    plt.savefig(f'./results/img/{os.path.splitext(os.path.basename(file_path))[0]}.png')
    plt.show()


def plot_sw(file_path, window_size):
    rewards = np.load(file_path)
    moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure()
    plt.plot(rewards, label='Social Welfare', alpha=0.3)
    plt.plot(moving_average, label=f'Moving Average ({window_size})')
    plt.legend()
    plt.xlabel('Update Step')
    plt.ylabel('Social Welfare')
    plt.savefig(f'./results/img/{os.path.splitext(os.path.basename(file_path))[0]}.png')
    plt.show()


def plot_incentive_awarness(csv_file_path):
    df = pd.read_csv(csv_file_path)
    steps = df['update step']
    incentive_awareness = df['incentive awareness']

    plt.figure()
    plt.scatter(steps, incentive_awareness, alpha=0.5, s=1)
    plt.xlabel('Step')
    plt.ylabel('Incentive Awareness')
    plt.title('Incentive Awareness across Steps (Scatter)')
    plt.show() 


def plot_tx_fee_vs_private_value(csv_file_path, step):
    df = pd.read_csv(csv_file_path)
    episode_data = df[df['update step']==step]
    private_value = episode_data['private value']
    tx_fee = episode_data['tx fee']

    plt.figure()
    plt.scatter(private_value, tx_fee, s=10, alpha=0.8, label='Private Value vs Transaction Fee')
    plt.xlabel('Private Value')
    plt.ylabel('Transaction Fee')
    plt.show()


def plot_strategies():
    import torch
    from agents.es import Net

    device = torch.device('cpu')
    strategies = Net(num_agents=1, num_actions=1).to(device)
    strategies.load_state_dict(torch.load(r'./results/models/ES_1.0_no_None.pth'))
    # print(strategies.nu)

    x = np.logspace(np.log10(1e-8), 0, 1000)
    y = strategies(torch.FloatTensor(x).to(torch.float64)\
        .reshape(-1, 1).to(device)).squeeze().detach().cpu().numpy()

    plt.figure()
    plt.plot(x, x, color='red', alpha=.5, label='Truthful')
    # plt.plot(x, np.zeros_like(x), color='green', alpha=.5, label='Zero')
    plt.plot(x, y, color='blue', label='Strategy')
    plt.title('Fee - Valuation')
    plt.xlabel('Valuation')
    plt.ylabel('Transaction Fee')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'./results/img/ES_1.0_no_None.png')
    plt.show()


if __name__ == '__main__':
    plot_strategies()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file_path', type=str, default=None, help='File Path')
    # parser.add_argument('--graph', type=str, default=None, help='Graph')
    # parser.add_argument('--window_size', type=int, default=100, help='Window Size')
    # parser.add_argument('--step', type=int, default=None, help='Step')
    # parser.add_argument('--label', type=str, default=None, help='Label')
    # args = parser.parse_args()

    # if args.graph == 'sw':
    #     plot_sw(args.file_path, args.window_size)
    # elif args.graph == 'incentive_awareness':
    #     plot_incentive_awarness(args.file_path)
    # elif args.graph == 'tx_fee_vs_private_value':
    #     plot_tx_fee_vs_private_value(args.file_path, args.step)
    # elif args.graph == 'plot':
    #     plot(args.file_path, args.label)
    # else:
    #     raise NotImplementedError
