import numpy as np
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines

MODELS = ['skip20', 'skip40', 'skip60', 'skip80', 'skip100',
          'no_sampling+BB', 'IHT+RF', 'no_sampling+CSP', 'NCR+DT', 'no_sampling+LOF', 'Accuracy=1']
ITEMS = ['SAVE_CI_master_sum', 'SAVE_wait_avg']
LABELS = ['skip20', 'skip40', 'skip60', 'skip80', 'skip100',
          'BB', 'IHT+RF', 'CSP', 'NCR+DT', 'LOF', 'ideal']

def cal_avg(project):
    # 1. read file
    input_path = f'../comparison/{project}.xlsx'
    df = pd.read_excel(input_path, sheet_name='all')

    # 2. calculate
    columns = ['model', 'SAVE_CI_master_sum', 'SAVE_One_CI', 'SAVE_wait_avg']
    df_result = pd.DataFrame(columns=columns)
    models = ['skip20', 'skip40', 'skip60', 'skip80', 'skip100',
              'IHT+RF', 'no_sampling+BB', 'no_sampling+CSP', 'NCR+DT', 'no_sampling+LOF']
    for m in models:
        # 2.1 get all items of certain model
        temp_df = df[df['model'] == m]
        # 2.2 calculate the average value of 'SAVE_CI_master_sum', 'SAVE_One_CI', and 'SAVE_wait_avg'
        save_ci_master_sum = "{:.1f}%".format(temp_df['SAVE_CI_master_sum'].mean() * 100)
        save_one_ci = "{:.1f}%".format(temp_df['SAVE_One_CI'].mean() * 100)
        save_wait_avg = "{:.1f}%".format(temp_df['SAVE_wait_avg'].mean()*100)
        temp_dict = {
            "model": m,
            "SAVE_CI_master_sum": save_ci_master_sum,
            "SAVE_One_CI": save_one_ci,
            "SAVE_wait_avg": save_wait_avg
        }
        df_result = pd.concat([df_result, pd.DataFrame(temp_dict, index=[0])])

    # 3. save to excel
    writer = pd.ExcelWriter(input_path, mode='a')
    df_result.to_excel(writer, sheet_name='avg', index=False)
    writer.close()

def draw_box_figure(programs, data, item, models=None):
    if models is None:
        models = MODELS
    fig, axes = plt.subplots(figsize=(22, 10), nrows=3, ncols=2)
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    positions = list(range(1, models.__len__()+1))
    for i in range(1, positions.__len__()):
        positions[i] += 0.8*i
    print(f'positions: {positions}')
    for i in range(6):
        row = int(i / 2)
        col = i % 2
        ax = axes[row, col]
        ax.boxplot(data[i], positions=positions, labels=models, widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor='#FFFFFF', color='black'),
                   medianprops=dict(color='#FF7F0F'),
                   showmeans=True, meanline=dict(linestyle='--'), meanprops=dict(color='green'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   flierprops=dict(marker='o', markersize=5, alpha=0.75))

        for j in range(data[i].__len__()):
            median = np.median(data[i][j])
            mean = np.mean(data[i][j])
            x = positions[j]+0.24
            y = (median + mean) / 2
            d = abs(median - mean)
            q1 = np.percentile(data[i][j], 25)
            q2 = np.percentile(data[i][j], 75)
            d = q2 - q1
            if i == 1:
                q2 = q2 + 0.05
                q1 = q1 - 0.05
            if i in [3, 4, 5] and j == 1:
                q2 = q2 + 0.04
                q1 = q1 - 0.04
            y1, y2 = q2, q1

            if median > mean:
                value_1, value_2 = "{:.1%}".format(median), "{:.1%}".format(mean)
                color_1, color_2 = '#FF7F0F', 'green'
            else:
                value_1, value_2 = "{:.1%}".format(mean), "{:.1%}".format(median)
                color_1, color_2 = 'green', '#FF7F0F'
            ax.text(x, y1, value_1, color=color_1, fontsize=12)
            ax.text(x, y2, value_2, color=color_2, fontsize=12)

        ax.set_xticks(positions)
        ax.set_xlim(0, 20.6)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_title(f'Project {programs[i]}', y=-0.28, fontsize=15, fontweight='600')
        ax.grid(ls='--')

    median_legend = mlines.Line2D([], [], color='#FF7F0F', label='Median')
    mean_legend = mlines.Line2D([], [], color='green', linestyle='--', label='Mean')
    ax = axes[0, 0]
    ax.legend(handles=[median_legend, mean_legend], loc='upper left')

    plt.savefig(f'./figs/project_{item}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def get_data_lists(project, item):
    # 1. read file
    filepath = f'../comparison/{project}.xlsx'
    df = pd.read_excel(filepath, sheet_name='data')

    # 2. get each model's data item list
    data = []
    for m in MODELS:
        df_m = df[df['model'] == m]
        data_m = df_m[item].tolist()
        data.append(data_m)
    return data

def get_avg_list(project, item):
    # 1. read file
    filepath = f'../comparison/{project}.xlsx'
    df = pd.read_excel(filepath, sheet_name=item)

    # 2. get project's data item list
    data = df[project].tolist()
    return data


if __name__ == "__main__":
    projects = ['A', 'B', 'C', 'D', 'E', 'F']
    for p in projects:
        cal_avg(p)
    for i in ITEMS:
        data_all = []
        data_avg = []
        for p in projects:
            temp_data = get_data_lists(p, i)
            data_all.append(temp_data)
            temp_avg = get_avg_list(p, i)
            data_avg.append(temp_avg)
        draw_box_figure(programs=projects, data=data_all, item=i, models=LABELS)
