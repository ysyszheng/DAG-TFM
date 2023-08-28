import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, default=None)
    parser.add_argument('--graph', type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.fn)
    plt.figure(figsize=(18, 6))

    if args.graph == 'scatter':
      steps = df['step']
      incentive_awareness = df['incentive awareness']

      plt.scatter(steps, incentive_awareness, alpha=0.5, s=1)
      plt.xlabel('Step')
      plt.ylabel('Incentive Awareness')
      plt.title('Incentive Awareness across Steps (Scatter)')
      plt.show() 
    elif args.graph == 'box':
      grouped_data = df.groupby('step')['incentive awareness'].apply(list).reset_index()

      plt.boxplot(grouped_data['incentive awareness'], labels=grouped_data['step'], showfliers=False)
      plt.xlabel('Step')
      plt.ylabel('Incentive Awareness')
      plt.title('Incentive Awareness across Steps (Boxplot)')
      plt.show()