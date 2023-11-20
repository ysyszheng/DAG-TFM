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

    frequencies, bins, _ = plt.hist(fee_data_filtered, bins=int(np.sqrt(len(fee_data_filtered))), color='blue', alpha=0.7, density=True)
    plt.xlabel('Fee Amount')
    plt.ylabel('Frequency')
    plt.title('Fee Distribution')
    # plt.grid(True)

    max_frequency = max(frequencies)
    max_frequency_index = np.argmax(frequencies)
    corresponding_bin = bins[max_frequency_index]
    print(corresponding_bin)
    plt.ylim(0, max_frequency * 1.1)
    plt.savefig(r'./assets/hist.png')
    plt.show()

def transform_fee_distribution():
    # * mean: 7167.122512974324, std: 38358.98737426391
    fee_data_file = os.path.join(DATA_FOLDER, 'fee.npy')
    fee_data = np.load(fee_data_file)
    mean_value = np.mean(fee_data)
    std_deviation = np.std(fee_data)
    normalized_data = (fee_data - mean_value) / std_deviation
    normalized_data = normalized_data[normalized_data <= 10]
    print(f'mean: {mean_value}, std: {std_deviation}')
    # normalized_data = np.log1p(fee_data)

    plt.hist(normalized_data, bins=50, alpha=0.7, density=True)
    plt.xlabel('Fee Amount')
    plt.ylabel('Frequency')
    plt.title('Fee Distribution (Values)')
    plt.grid(True)
    plt.show()


def fit_fee_distribution():
    # * f(x) = a * x ** (-k), x >= x_min
    # * f(x) = 0.12628989798860135 * x ^ (-1.1262898979886014), x >= 1.0
    fee_data_file = os.path.join(DATA_FOLDER, 'fee.npy')
    fee_data = np.load(fee_data_file)
    fee_data_filtered = fee_data[fee_data > 0]

    # * fit the data to power law distribution
    # * maximum likelihood estimation
    fee_min = np.min(fee_data_filtered)
    k = len(fee_data_filtered) / np.sum(np.log(fee_data_filtered / fee_min)) + 1
    a = (k - 1)/ (fee_min ** (1 - k))
    print(f'k: {k}, a: {a}, fee_min: {fee_min}, f(x) = {a} * x ^ (-{k})')
    legend_label = r'$\mathrm{Fitted\ Power\ Law\ PDF:}$' + '\n' + r'$f(x) = %.4f \cdot x^{%.4f}$' % (a, -k)

    # * plot the fitted distribution
    threshold = 40000
    fee_data_filtered = fee_data_filtered[fee_data_filtered < threshold]
    plt.hist(fee_data_filtered, bins=50, alpha=0.7, density=True, label='Histogram')

    x = np.linspace(fee_min, threshold, 1000)
    fitted_pdf = a * x ** (-k)
    plt.plot(x, fitted_pdf, 'r-', lw=2, label=legend_label)

    plt.xlabel('Fee Amount')
    plt.ylabel('Probability Density')
    plt.title('Fitted Power Law Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # get_fee_each_month()
    # get_fee()
    plot_fee_distribution()
    # transform_fee_distribution()
    # fit_fee_distribution()
