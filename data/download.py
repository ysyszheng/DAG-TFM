import concurrent.futures
import requests
import pandas as pd

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://gz.blockchair.com/bitcoin/transactions/',
}

# 定义爬取并保存数据的函数
def scrape_and_save_data(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        filename = url.split('/')[-1]
        with open(filename, 'wb') as file:
            file.write(response.content)
            print(f"已保存文件: {filename}")
    else:
        print(f"下载失败: {url}")

base_url = "https://gz.blockchair.com/bitcoin/transactions/blockchair_bitcoin_transactions_"
date_format = "%Y%m%d"
start_date = "20230401"
end_date = "20230705"

# 生成日期范围
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# 创建线程池，最大并发数为 10
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# 并行爬取数据
results = []
for date in date_range:
    url = base_url + date.strftime(date_format) + ".tsv.gz"
    future = executor.submit(scrape_and_save_data, url)
    results.append(future)

# 等待所有任务完成
concurrent.futures.wait(results)

print("数据爬取完成！")
