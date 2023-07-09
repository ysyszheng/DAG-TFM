import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./sqrt.csv', header=None)

# episode_ranges = [(i, i+100) for i in range(0, 2000, 100)]
episode_ranges = [(2000,2001)]

# colors = plt.cm.rainbow(np.linspace(0, 1, len(episode_ranges)))
colors = plt.cm.tab10(np.linspace(0, 1, len(episode_ranges)))

plt.figure(figsize=(8, 6))

for i, (start, end) in enumerate(episode_ranges):
    episode_data = data[(data[0] >= start) & (data[0] < end)]
    # episode_data = data[data[0]==400]
    private_value = episode_data[1]
    transaction_fee = episode_data[2]
    probs = episode_data[4]
    cnt = episode_data[6]
    
    plt.scatter(private_value, cnt)

plt.xlabel('Private Value')
plt.ylabel('Number of times being packed')
# plt.title('Private Value vs Transaction Fee by Episode Ranges')
plt.savefig('./img/private_value_vs_count.png')
plt.show()

