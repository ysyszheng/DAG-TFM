import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.optimize import curve_fit


DATA_FOLDER = './data'


def get_fee_each_month():
    tsv_files = [file for file in os.listdir(DATA_FOLDER) if file.endswith('.tsv')]

    monthly_data = {}

    for tsv_file in tsv_files:
        file_path = os.path.join(DATA_FOLDER, tsv_file)
        
        year_month = tsv_file.split('.')[0][:6]
        
        df = pd.read_csv(file_path, sep='\t')
        
        column_16_data = df.iloc[:, 15].values # column 16 is the fee column
        
        if year_month not in monthly_data:
            monthly_data[year_month] = column_16_data
        else:
            monthly_data[year_month] = np.concatenate((monthly_data[year_month], column_16_data))

    for year_month, data in monthly_data.items():
        print(year_month)
        print(data)
        output_file = os.path.join(DATA_FOLDER, f'fee_{year_month}.npy')
        np.save(output_file, data)

def get_fee():
    fee_files = [file for file in os.listdir(DATA_FOLDER) if file.startswith('fee_') and file.endswith('.npy')]

    combined_data = np.array([], dtype=float)

    for fee_file in fee_files:
        file_path = os.path.join(DATA_FOLDER, fee_file)
        data = np.load(file_path)
        combined_data = np.concatenate((combined_data, data))

    combined_data = np.array(combined_data, dtype=float)

    output_file = os.path.join(DATA_FOLDER, 'fee.npy')
    np.save(output_file, combined_data)

def plot_fee_distribution():
    fee_data_file = os.path.join(DATA_FOLDER, 'fee.npy')
    fee_data = np.load(fee_data_file)
    fee_data_filtered = fee_data[fee_data < 40000]

    plt.hist(fee_data_filtered, bins=50, alpha=0.7, density=True)
    plt.xlabel('Fee Amount')
    plt.ylabel('Frequency')
    plt.title('Fee Distribution (Values < 40000)')
    plt.grid(True)
    plt.show()


def fit_fee_distribution():
    fee_data_file = os.path.join(DATA_FOLDER, 'fee.npy')
    fee_data = np.load(fee_data_file)
    fee_data_filtered = fee_data[fee_data < 40000]

    params = expon.fit(fee_data)
    print(params)

    x = np.linspace(0, 40000, 100000)
    fitted_pdf = expon.pdf(x, *params)

    plt.hist(fee_data_filtered, bins=50, alpha=0.7, density=True, label='Histogram')

    plt.plot(x, fitted_pdf, 'r-', lw=2, label='Fitted Exponential PDF')

    plt.xlabel('Fee Amount')
    plt.ylabel('Probability Density')
    plt.title('Fitted Exponential Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # get_fee_each_month()
    # get_fee()
    # plot_fee_distribution()
    # fit_fee_distribution()
    fee_data_file = os.path.join(DATA_FOLDER, 'fee.npy')
    fee_data = np.load(fee_data_file)
    fee_data_filtered = np.log(fee_data[fee_data > 0])
    mean_value = np.mean(fee_data_filtered)
    std_deviation = np.std(fee_data_filtered)
    normalized_data = (fee_data_filtered - mean_value) / std_deviation

    plt.hist(normalized_data, bins=50, alpha=0.7, density=True)
    plt.xlabel('Fee Amount')
    plt.ylabel('Frequency')
    plt.title('Fee Distribution (Values)')
    plt.grid(True)
    plt.show()
