from web3 import Web3
import json
import csv
import concurrent.futures

# Connect to full nodes
w3 = Web3(Web3.HTTPProvider('https://eth-mainnet.g.alchemy.com/v2/46imDhxi2dnWFNWl5cNirB2_fY0u_Wxa'))

# 定义保存结果的函数
def save_result(result):
    with open('tx_fees.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)

# 定义爬取单个区块的函数
def scrape_block(block_number):
    block = w3.eth.get_block(block_number)
    # 提取区块中的交易费数据
    # transaction_fees = [w3.eth.get_transaction(tx)["gasPrice"] for tx in block.transactions]
    # print(transaction_fees)
    # print(block.transactions)
    for tx in block.transactions:
        print(block_number, ",", w3.eth.get_transaction(tx)["gasPrice"])
    # 保存结果到文件
    # for fee in transaction_fees:
    #     print(fee)
    #     save_result([block_number, fee])

start_block = 16384919  # 起始区块号
end_block = 17634886  # 结束区块号

# 创建线程池，最大并发数为 10
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# 并行爬取数据
results = []
for block_number in range(start_block, end_block+1):
    future = executor.submit(scrape_block, block_number)
    results.append(future)

# 等待所有任务完成
concurrent.futures.wait(results)

print("数据爬取完成！")