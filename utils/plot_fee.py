# import matplotlib.pyplot as plt
# import pickle
# import numpy as np

# with open('fee.pkl', 'rb') as f:
#     fee_list = pickle.load(f)

# # 绘制直方图
# plt.figure(figsize=(8, 6))
# plt.hist(fee_list, bins=10, density=True, cumulative=True, label='Empirical CDF')
# plt.xlabel('Transaction Fee')
# plt.ylabel('CDF')
# plt.title('Empirical CDF of Transaction Fee')
# plt.savefig('./img/empirical_cdf.png')
# plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv('./data/20230101.tsv', sep='\t', header=0)
counts, bins = np.histogram(data['fee'])
plt.stairs(counts, bins)
plt.show()
