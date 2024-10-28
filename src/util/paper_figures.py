import os
import glob
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import pyplot as plt


def metric_table(location, metrics, months, analysis):
    folder_path = os.path.abspath(os.path.join(os.getcwd(), '../results/' + location + '/'))
    file_paths = glob.glob(os.path.join(folder_path, '*'))
    file_paths = [item for item in file_paths if analysis in item]
    files = [os.path.basename(file_path) for file_path in file_paths]

    row_names = []
    for file_name in files:
        name = file_name.replace('summary_table_', '').rstrip('.csv')
        row_names.append(name)

    column_names = ['Baseline RMSE', 'Target RMSE', 'Transfer RMSE', 'Baseline MBE', 'Target MBE', 'Transfer MBE',
                    'Baseline MAE', 'Target MAE', 'Transfer MAE']

    # Set up a multi index with the site name and the MSE metrics
    multi_index = pd.MultiIndex.from_product([row_names, column_names], names=['Site', 'Metric'])

    # Set up the canvas for the dataframe
    summary_table = pd.DataFrame(index=multi_index, columns=months, dtype=float)

    # Input the data from all the csv files into the dataframe
    for i in range(len(file_paths)):
        data = pd.read_csv(file_paths[i], index_col=0).transpose()[0:9].values

        summary_table.loc[row_names[i]].iloc[:, :len(data[0])] = data

    base = summary_table.groupby('Metric').get_group(metrics[0]).T
    target = summary_table.groupby('Metric').get_group(metrics[1]).T
    transfer = summary_table.groupby('Metric').get_group(metrics[2]).T

    return base, target, transfer


def main_figure(base, target, transfer, metrics, months, quantile_min=0, quantile_max=1):
    target_max = target.quantile(quantile_max, interpolation='linear', axis=1)[0:len(months)]
    target_min = target.quantile(quantile_min, interpolation='linear', axis=1)[0:len(months)]
    transfer_max = transfer.quantile(quantile_max, interpolation='linear', axis=1)[0:len(months)]
    transfer_min = transfer.quantile(quantile_min, interpolation='linear', axis=1)[0:len(months)]

    # Create figure with larger size and higher DPI
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=180, facecolor='none')

    fontsize = 16

    # Plot target data
    ax1.plot(months, target.median(axis=1)[:len(months)], 'b', label="Median target")
    ax1.fill_between(months, target_max, target_min, alpha=0.1)

    # Plot transfer data
    ax1.plot(months, transfer.median(axis=1)[:len(months)], 'r', label="Median transfer")
    ax1.fill_between(months, transfer_max, transfer_min, alpha=0.1)

    # Plot baseline data
    ax1.plot(months, base.median(axis=1)[:len(months)], 'g', label="Median Naive")

    # Set labels and ticks
    ax1.set_xlabel('Months of data', fontsize=fontsize + 2)
    ax1.set_ylabel(metrics, fontsize=fontsize + 2)
    ax1.tick_params(axis='both', labelsize=fontsize)

    # Customize spines and background
    for spine in ax1.spines.values():
        spine.set_color('black')
    ax1.set_facecolor('white')
    ax1.set_xlim(months[0], months[-1])

    # Add legend with custom frame
    legend = ax1.legend(fontsize=16)
    legend.get_frame().set_facecolor('white')

    # Show plot
    plt.show()


def distance_figure(base, target, transfer, metrics, months, distances, quantile_min=0, quantile_max=1):
    # Calculate the median and the confidence intervals for each distance group
    median_dict = {}
    lower_bound_dict = {}
    upper_bound_dict = {}

    for dist in distances:
        dist_columns = [col for col in transfer.columns if col[0].endswith(dist)]
        median_dict[dist] = transfer[dist_columns].median(axis=1)
        lower_bound_dict[dist] = transfer[dist_columns].quantile(quantile_min, axis=1)  # 25th percentile
        upper_bound_dict[dist] = transfer[dist_columns].quantile(quantile_max, axis=1)  # 75th percentile

    # Create a color map from red to green
    cmap = cm.get_cmap('RdYlGn_r')
    all_medians = np.concatenate([median_dict[dist].values for dist in distances])
    norm = mcolors.Normalize(vmin=min(all_medians), vmax=max(all_medians))

    # Create a figure with a large size
    fig = plt.figure(figsize=(10, 5), dpi=180, facecolor='none')

    fontsize = 16

    ax1 = fig.add_subplot(111)

    # Store the lines for the legend
    lines_for_legend = []

    for dist in distances:
        color = cmap(norm(median_dict[dist].mean()))
        line, = ax1.plot(months, median_dict[dist], label=dist, color=color)
        ax1.fill_between(months, lower_bound_dict[dist], upper_bound_dict[dist], color=color, alpha=0.3)
        lines_for_legend.append(line)

    # Plot the median Naive RMSE
    median_naive_rmse, = ax1.plot(months, base.median(axis=1)[0:len(months)], 'black', alpha=1,
                                  label="Median Naive")
    lines_for_legend.append(median_naive_rmse)

    ax1.plot(months, target.median(axis=1)[0:len(months)], 'b', label="Median target", alpha=1)

    ax1.set_xlabel('Months of data', fontsize=fontsize + 2)  # Increase the font size
    ax1.set_ylabel(metrics, fontsize=fontsize + 2)  # Increase the font size
    ax1.tick_params(axis='both', labelsize=fontsize)  # Adjust the font size as needed

    # Customize spines and background
    for spine in ax1.spines.values():
        spine.set_color('black')
    ax1.set_facecolor('white')
    ax1.set_xlim(months[0], months[-1])

    distances = [s.lstrip('_') for s in distances]
    legend_names = distances + ['Median Naive']

    legend = ax1.legend(lines_for_legend, legend_names, fontsize=16, ncol=2)
    frame = legend.get_frame()
    frame.set_facecolor('white')

    # Show the figure
    plt.title('Error given the months of data available in the target domain')
    plt.show()
