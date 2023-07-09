import os
import pandas as pd

folder_path = './data'  # 将路径替换为实际的文件夹路径
column_index = 15

data_list = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.tsv'):
        file_path = os.path.join(folder_path, file_name)
        print(f"正在读取文件: {file_path}")
        # 使用read_csv函数读取tsv文件，设置分隔符为制表符，并指定第一行为表头
        df = pd.read_csv(file_path, sep='\t', header=0)
        
        # 提取第15列数据
        if len(df.columns) > column_index:
            column_name = df.columns[column_index]
            data = df[column_name].tolist()
            data_list.extend(data)

# 打印第15列数据
print(data_list)
