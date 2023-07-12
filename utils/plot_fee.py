import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv('./data/20230101.tsv', sep='\t', header=0)
counts, bins = np.histogram(data['fee'])
plt.stairs(counts, bins)
plt.show()
