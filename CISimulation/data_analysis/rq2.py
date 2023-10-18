import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

PROJECTS = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip',
            'akka@akka', 'opf@openproject']
PROJECTS_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F']
X_LABELS = ['interval_mean', 'fail_rate', 'avg_ci_execution_time']
X_NAMES = ['the frequency of commits', 'the fail rate of CI', 'the average time cost of executing CI']
MODEL = ['LOF', 'NCR+DT', 'LOF', 'NCR+DT', 'NCR+DT', 'NCR+DT']  # best model of each project
SKIP = 'skip100'

def get_data_with_all(project: str, model: str):
    # read file
    filepath = f'./comparison/{PROJECTS_ALPHABET[PROJECTS.index(project)]}.xlsx'
    df = pd.read_excel(filepath)

    # filter dataframe according to model
    df = df[df['model'] == model]
    # get columns [x (interval_mean, fail_rate, or avg_execution_time), SAVE_ci_master_sum, SAVE_wait_avg]
    df = df[['interval_mean', 'fail_rate', 'avg_ci_execution_time', 'SAVE_CI_master_sum', 'SAVE_wait_avg']]
    return df

def cal_r_squared(y: list, y_pred: list) -> float:
    # R^2 = 1 - SS_regression / SS_total
    # SS_regression = \sum(y - y_regression)^2
    # SS_total = \sum(y - y_mean)^2
    res_y = np.array(y) - np.array(y_pred)
    ss_res = np.sum(res_y ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def draw_heat_map(projects=None):
    if projects is None:
        projects = PROJECTS
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(20, 25))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for i in range(projects.__len__()):
        ax_model = axes[i, 0]
        ax_skip = axes[i, 1]

        df_model = get_data_with_all(project=projects[i], model=MODEL[i]).fillna(0)
        df_skip = get_data_with_all(project=projects[i], model=SKIP).fillna(0)

        x = ['interval_mean', 'fail_rate', 'avg_execution_time']
        y = [r'$S_{ci\_m}(M, O)$', r'$S_{wait}(M, O)$']

        temp_model = df_model.corr(method='pearson')
        temp_skip = df_skip.corr(method='pearson')

        corr_c, p_c = pearsonr(df_model['interval_mean'], df_model['SAVE_wait_avg'])
        corr_f, p_f = pearsonr(df_model['fail_rate'], df_model['SAVE_wait_avg'])
        corr_e, p_e = pearsonr(df_model['avg_ci_execution_time'], df_model['SAVE_wait_avg'])
        print(f'=== Project: {projects[i]}')
        print(f'corr_c: {corr_c}, p_c: {p_c}')
        print(f'corr_f: {corr_f}, p_c: {p_c}')
        print(f'corr_e: {corr_e}, p_e: {p_e}')
        # matrix
        m_list1 = temp_model.iloc[0, 3:].tolist()
        m_list2 = temp_model.iloc[1, 3:].tolist()
        m_list3 = temp_model.iloc[2, 3:].tolist()
        matrix_m = [m_list1, m_list2, m_list3]
        corr_matrix_m = pd.DataFrame(matrix_m)

        sns.heatmap(corr_matrix_m, ax=ax_model, cmap='coolwarm', xticklabels=y, yticklabels=x, annot=True)

        s_list1 = temp_skip.iloc[0, 3:].tolist()
        s_list2 = temp_skip.iloc[1, 3:].tolist()
        s_list3 = temp_skip.iloc[2, 3:].tolist()
        matrix_s = [s_list1, s_list2, s_list3]
        corr_matrix_s = pd.DataFrame(matrix_s)
        sns.heatmap(corr_matrix_s, ax=ax_skip, cmap='coolwarm', xticklabels=y, yticklabels=x, annot=True)

        ax_model.set_title(f"{projects[i]}'s best model")
        ax_skip.set_title(f"{projects[i]}'s skip-100")
    plt.show()


if __name__ == "__main__":
    draw_heat_map()
