import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

data = pd.read_csv('./none.csv', header=None)

episode_data = data[data[0]==2000]
private_value = episode_data[1].tolist()
# probs = episode_data[5].tolist()

num_users = 40

probs = [10/num_users] * num_users

transactions = list(range(len(probs)))

user_transactions = []
for _ in range(num_users):
    user_transactions.extend(random.choices(transactions, probs, k=random.randint(0, len(transactions))))

unique_transactions = set(user_transactions)
num_unique_transactions = len(unique_transactions)

print(f"最终有 {num_unique_transactions} 个不同的交易被打包进入区块。")

total_private_value = sum(private_value[transaction_id] for transaction_id in unique_transactions)

print(f"被打包的交易的private value之和为: {total_private_value}")