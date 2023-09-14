import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import seaborn as sns

def plot_private_values(fn='./envs/fee.pkl', stop=40000, step=100):
    with open(fn, 'rb') as file:
        fee_list = pickle.load(file)
    # hist, bins = np.histogram(fee_list, bins=np.arange(start=0, stop=stop, step=step))
    # plt.figure(figsize=(18, 6))
    # plt.bar(bins[:-1], hist, width=bins[1]-bins[0])
    # x = np.arange(start=0, stop=stop, step=.1)
    # y = np.exp(15.357) * x ** (-2.411)
    # plt.plot(x, y, color='red', label='Fitted Power Law')
    # plt.savefig(f'./img/private_values.png')
    # plt.show()
    sns.set_palette("hls") #设置所有图的颜色，使用hls色彩空间
    fee_list = [fee for fee in fee_list if fee < 40000]
    sns.distplot(fee_list,color="r",bins=400,kde=True)
    plt.show()


def plot_rewards(steps, window_size):
    rewards = np.load(f'./rewards/rewards_{steps}.npy')
    moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(18, 6))
    plt.plot(rewards, label='Rewards')
    plt.plot(moving_average, label=f'Moving Average ({window_size})')
    plt.legend()
    plt.savefig(f'./img/rewards_{steps}.png')
    plt.show()


if __name__ == '__main__':
    # reward = np.load('./social_welfare_history.npy')
    # plt.figure(figsize=(18, 6))
    # plt.plot(reward, label='Rewards')
    # plt.legend()
    # plt.show()
    action = np.load('action_history.npy')
    print(action)
