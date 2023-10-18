import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

MEDIAN_VALUES = {
    'interval_mean': 37.2739877,
    'fail_rate': 0.14,
    'avg_ci_execution_time': 13.492
}

CURVES = ['accuracy', 'precision_passed', 'precision_failed', 'recall_passed', 'recall_failed',
          'f1_passed', 'f1_failed']
X_NAMES = [r'$Accuracy$', r'$Precision_{passed}$', r'$Precision_{failed}$', r'$Recall_{passed}$', r'$Recall_{failed}$',
           r'$F1_{passed}$', r'$F1_{failed}$']
a = 'interval_mean'
b = 'fail_rate'
c = 'avg_ci_execution_time'
TITLES = [f'{a}, {b}, and {c} are less than the median',
          f'{a} and {b} are less than the median, while {c} are greater than the median',
          f'{a} and {c} are less than the median, while {b} are greater than the median',
          f'{a} is less than the median, while {b} and {c} are greater than the median',
          f'{a} is greater than the median, while {b} and {c} are less than the median',
          f'{a} and {c} are greater than the median, while {b} are less than the median',
          f'{a} and {b} are greater than the median, while {c} is less than the median',
          f'{a}, {b}, and {c} are greater than the median']

'''
low: 0, high: 1
interval_mean           0   0   0   0   1   1   1   1
fail_rate               0   0   1   1   0   0   1   1
avg_execution_time      0   1   0   1   0   1   0   1
'''

def divide_group():
    df = pd.read_excel('./temp/final.xlsx', sheet_name='Sheet1')
    # low, low, low
    temp_1 = df[df['interval_mean'] <= MEDIAN_VALUES['interval_mean']]
    temp_2 = temp_1[temp_1['fail_rate'] <= MEDIAN_VALUES['fail_rate']]
    df_1 = temp_2[temp_2['avg_ci_execution_time'] <= MEDIAN_VALUES['avg_ci_execution_time']]

    # low, low, high
    temp_1 = df[df['interval_mean'] <= MEDIAN_VALUES['interval_mean']]
    temp_2 = temp_1[temp_1['fail_rate'] <= MEDIAN_VALUES['fail_rate']]
    df_2 = temp_2[temp_2['avg_ci_execution_time'] > MEDIAN_VALUES['avg_ci_execution_time']]

    # low, high, low
    temp_1 = df[df['interval_mean'] <= MEDIAN_VALUES['interval_mean']]
    temp_2 = temp_1[temp_1['fail_rate'] > MEDIAN_VALUES['fail_rate']]
    df_3 = temp_2[temp_2['avg_ci_execution_time'] <= MEDIAN_VALUES['avg_ci_execution_time']]

    # low, high, high
    temp_1 = df[df['interval_mean'] <= MEDIAN_VALUES['interval_mean']]
    temp_2 = temp_1[temp_1['fail_rate'] > MEDIAN_VALUES['fail_rate']]
    df_4 = temp_2[temp_2['avg_ci_execution_time'] > MEDIAN_VALUES['avg_ci_execution_time']]

    # high, low, low
    temp_1 = df[df['interval_mean'] > MEDIAN_VALUES['interval_mean']]
    temp_2 = temp_1[temp_1['fail_rate'] <= MEDIAN_VALUES['fail_rate']]
    df_5 = temp_2[temp_2['avg_ci_execution_time'] <= MEDIAN_VALUES['avg_ci_execution_time']]

    # high, low, high
    temp_1 = df[df['interval_mean'] > MEDIAN_VALUES['interval_mean']]
    temp_2 = temp_1[temp_1['fail_rate'] <= MEDIAN_VALUES['fail_rate']]
    df_6 = temp_2[temp_2['avg_ci_execution_time'] > MEDIAN_VALUES['avg_ci_execution_time']]

    # high, high, low
    temp_1 = df[df['interval_mean'] > MEDIAN_VALUES['interval_mean']]
    temp_2 = temp_1[temp_1['fail_rate'] > MEDIAN_VALUES['fail_rate']]
    df_7 = temp_2[temp_2['avg_ci_execution_time'] <= MEDIAN_VALUES['avg_ci_execution_time']]

    # high, high, high
    temp_1 = df[df['interval_mean'] > MEDIAN_VALUES['interval_mean']]
    temp_2 = temp_1[temp_1['fail_rate'] > MEDIAN_VALUES['fail_rate']]
    df_8 = temp_2[temp_2['avg_ci_execution_time'] > MEDIAN_VALUES['avg_ci_execution_time']]

    df_list = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8]
    return df_list

def cal_r_squared(y: list, y_pred: list) -> float:
    # R^2 = 1 - SS_regression / SS_total
    # SS_regression = \sum(y - y_regression)^2
    # SS_total = \sum(y - y_mean)^2
    res_y = np.array(y) - np.array(y_pred)
    ss_res = np.sum(res_y ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def draw_heat_map():
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(20, 25))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    df_list = divide_group()
    for i in range(8):
        ax = axes[i]
        df = df_list[i]
        df = df[['accuracy', 'precision_passed', 'precision_failed', 'recall_passed', 'recall_failed',
                 'f1_passed', 'f1_failed', 'SAVE_CI_master_sum', 'SAVE_wait_avg']]
        temp = df.corr(method='pearson')

        x = ['accuracy', 'precision_passed', 'precision_failed', 'recall_passed', 'recall_failed',
             'f1_passed', 'f1_failed']
        y = ['SAVE_CI_master_sum', 'SAVE_wait_avg']

        print(f'=== {TITLES[i]} ===')
        for j in range(7):
            corr_j, p_j = pearsonr(df[x[j]], df[y[1]])
            print(f'corr_{x[j]}: {corr_j}, p_{x[j]}: {p_j}')

        matrix = []
        for i in range(7):
            matrix.append(temp.iloc[i, 7:].tolist())
        new_matrix = [[0 for _ in range(7)] for _ in range(2)]
        for m in range(7):
            for n in range(2):
                new_matrix[n][m] = matrix[m][n]

        corr_matrix = pd.DataFrame(new_matrix)
        sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', xticklabels=x, yticklabels=y, annot=True)
        ax.set_title(f'{TITLES[i]}')
    plt.show()


if __name__ == "__main__":
    draw_heat_map()
