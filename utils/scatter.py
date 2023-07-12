import pandas as pd
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type = str, default = None, help = 'File Name')
    parser.add_argument('--step', type = int, default = None, help = 'Step')
    args = parser.parse_args()

    data = pd.read_csv(args.file)
    episode_data = data[data['step']==args.step]
    private_value = episode_data['private value']
    transaction_fee = episode_data['transaction fee']
    probs = episode_data['probability']

    plt.figure(figsize=(8, 6))
    plt.scatter(private_value, transaction_fee, s=10, alpha=0.8, label='Private Value vs Transaction Fee')
    plt.xlabel('Private Value')
    plt.ylabel('Transaction Fee')
    plt.savefig('./img/private_value_vs_count.png')
    plt.show()
