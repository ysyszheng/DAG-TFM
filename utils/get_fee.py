import os
import pandas as pd

folder_path = './data'
column_index = 15

data_list = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.tsv'):
        file_path = os.path.join(folder_path, file_name)
        print(f"正在读取文件: {file_path}")
        df = pd.read_csv(file_path, sep='\t', header=0)
        
        # 提取第15列数据
        if len(df.columns) > column_index:
            column_name = df.columns[column_index]
            data = df[column_name].tolist()
            data_list.extend(data)

print(data_list)
