import pandas as pd
import numpy as np
import os

def process_sim_data(project, type, models):
    temp_path = f'./sim_data/{type}/temp_result'
    final_path = f'./sim_data/{type}/final_result'
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    if not os.path.exists(final_path):
        os.mkdir(final_path)

    input_path = f'./sim_data/{type}/{project}_{type}.xlsx'
    sim_df = pd.read_excel(input_path)
    sim_df['Value'].fillna(0, inplace=True)
    data = sim_df['Value'].tolist()

    total_situation = len(models)  # the number of different PCI strategies
    each_situation_round = 100  # the number of executions of each model
    step = 31  # the number of metrics output from simulation
    index, round = 0, 0
    column = ['conf_num', 'resource_num', 'model',
              'total_commits', 'now_commits', 'skip_commits',
              'now_wait_avg', 'skip_wait_avg', 'first_CI_avg',
              'resource_load_ratio',
              'master', 'other', 'TP', 'FP', 'TN', 'FN',
              'CI_master_sum', 'CI_other_sum', 'wait_avg', 'fault_location_sum',
              'waitInterval_sum', 'waitInterval_avg']

    df_final_result = pd.DataFrame(columns=column)

    for conf_num in range(30):
        print(f'{project} config number: {conf_num}')
        output_path_temp = f'./sim_data/{type}/temp_result/{project}_{conf_num}_{type}_temp_result.xlsx'
        writer_temp = pd.ExcelWriter(output_path_temp)
        for i in range(total_situation):
            df_temp_result = pd.DataFrame(columns=column)
            for j in range(each_situation_round):
                total_commits = data[index + 1]
                if total_commits == 0:
                    continue
                now_commits, skip_commits = data[index + 6], data[index + 8] + data[index + 10]
                resource_num, resource_load_ratio = data[index + 3], data[index + 4]
                now_wait_avg = 0 if now_commits == 0 else data[index + 5] / now_commits
                skip_wait_avg = 0 if skip_commits == 0 else (data[index + 7] + data[index + 9]) / skip_commits
                wait_avg = (data[index + 5] + data[index + 7] + data[index + 9]) / total_commits
                first_CI_avg = (data[index + 11] + data[index + 13] + data[index + 15]) / total_commits
                master, other = data[index + 25], data[index + 26]
                TP, FP, TN, FN = data[index + 17], data[index + 18], data[index + 19], data[index + 20]
                CI_master_sum, CI_other_sum = data[index + 21], data[index + 23]
                fault_location_sum, waitInterval_sum = data[index + 27], data[index + 29]
                try:
                    waitInterval_avg = waitInterval_sum / data[index + 30]
                except:
                    waitInterval_avg = 0
                dic_temp = {'resource_num': resource_num, 'model': models[i], 'total_commits': total_commits,
                            'now_commits': now_commits, 'skip_commits': skip_commits, 'now_wait_avg': now_wait_avg,
                            'skip_wait_avg': skip_wait_avg, 'first_CI_avg': first_CI_avg,
                            'resource_load_ratio': resource_load_ratio, 'master': master, 'other': other,
                            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
                            'CI_master_sum': CI_master_sum, 'CI_other_sum': CI_other_sum,
                            'wait_avg': wait_avg, 'fault_location_sum': fault_location_sum,
                            'waitInterval_sum': waitInterval_sum, 'waitInterval_avg': waitInterval_avg,
                            'config_num': conf_num}
                df_temp_result = df_temp_result.append(dic_temp, ignore_index=True)
                index += step
                round += 1
            df_temp_result.to_excel(writer_temp, sheet_name=models[i], index=False)

            total_commits = df_temp_result['total_commits'].mean()
            now_commits = df_temp_result['now_commits'].mean()
            skip_commits = df_temp_result['skip_commits'].mean()
            now_wait_avg = df_temp_result['now_wait_avg'].mean()
            skip_wait_avg = df_temp_result['skip_wait_avg'].mean()
            first_CI_avg = df_temp_result['first_CI_avg'].mean()
            resource_load_ratio = df_temp_result['resource_load_ratio'].mean()

            master, other = df_temp_result['master'].mean(), df_temp_result['other'].mean()
            TP, FP = df_temp_result['TP'].mean(), df_temp_result['FP'].mean()
            TN, FN = df_temp_result['TN'].mean(), df_temp_result['FN'].mean()
            CI_master_sum = df_temp_result['CI_master_sum'].mean()
            CI_other_sum = df_temp_result['CI_other_sum'].mean()
            wait_avg = df_temp_result['wait_avg'].mean()
            fault_location_sum = df_temp_result['fault_location_sum'].mean()
            waitInterval_sum = df_temp_result['waitInterval_sum'].mean()
            waitInterval_avg = df_temp_result['waitInterval_avg'].mean()

            dic_final = {'conf_num': conf_num, 'resource_num': resource_num, 'model': models[i],
                         'total_commits': total_commits, 'now_commits': now_commits, 'skip_commits': skip_commits,
                         'now_wait_avg': now_wait_avg, 'skip_wait_avg': skip_wait_avg, 'first_CI_avg': first_CI_avg,
                         'resource_load_ratio': resource_load_ratio, 'master': master, 'other': other, 'TP': TP,
                         'FP': FP, 'TN': TN, 'FN': FN, 'CI_master_sum': CI_master_sum, 'CI_other_sum': CI_other_sum,
                         'wait_avg': wait_avg, 'fault_location_sum': fault_location_sum,
                         'waitInterval_sum': waitInterval_sum, 'waitInterval_avg': waitInterval_avg
                         }
            df_final_result = df_final_result.append(dic_final, ignore_index=True)
        writer_temp.close()
        output_path_final = f'./sim_data/{type}/final_result/{project}_{type}_final_result.xlsx'
        writer_final = pd.ExcelWriter(output_path_final)
        df_final_result.to_excel(writer_final, index=False)
        writer_final.close()

def compare(project):
    output_path = f'./comparison/{project}.xlsx'
    types = ['Without', 'Random', 'PCI_5models']
    columns = ['resource_num', 'model',
               'total_commits', 'now_commits', 'skip_commits',
               'now_wait_avg', 'skip_wait_avg', 'first_CI_avg',
               'resource_load_ratio', 'master', 'other',
               'TP', 'FP', 'TN', 'FN',
               'CI_master_sum', 'CI_other_sum', 'wait_avg', 'fault_location_sum',
               'waitInterval_sum', 'waitInterval_avg']
    df_e = pd.DataFrame(columns=columns)
    for i in range(30):
        for type in types:
            input_path = f'./sim_data/{type}/final_result/{project}_{type}_final_result.xlsx'
            df = pd.read_excel(input_path)
            df = df[df['conf_num'] == i]
            df = df[columns]
            df_e = df_e.append(df, ignore_index=True)
    df_e.to_excel(output_path, index=False, header=True)

def merge_result(program):
    input_path = f'./comparison/{program}.xlsx'
    total_situations = 9   # the number of different simulation situations
    datas = np.array([[] for _ in range(total_situations * 30)])
    df = pd.read_excel(input_path)
    df['One_CI'] = df['CI_master_sum'] - df['fault_location_sum']
    d = df[['resource_num', 'model', 'waitInterval_avg', 'fault_location_sum', 'CI_master_sum', 'One_CI',
            'wait_avg']].values
    data = np.hstack((datas, d))
    index = 0
    save = []
    for k in range(30):
        base = data[index]
        save.append([np.nan] * (len(base) - 5))
        for i in range(index + 1, index + 2):
            row = []
            for j in range(5, len(base)):
                s = np.nan if base[j] == 0 else (base[j] - data[i][j]) / base[j]
                row.append(s)
            save.append(row)
        index += total_situations
    save = np.array(save)
    new_save = []
    for s in save:
        new_save.append(np.array(s))
    save = np.array(new_save)
    print(save.shape)
    print(data.shape)
    data = np.hstack((data, save))

    columns = ['conf_num', 'resource_num', 'model', 'waitInterval_avg', 'fault_location_sum',
               'CI_master_sum', 'One_CI', 'wait_avg',
               'SAVE_CI_master_sum', 'SAVE_One_CI', 'SAVE_wait_avg']
    df = pd.DataFrame(data=data, columns=columns)
    writer = pd.ExcelWriter(input_path, mode='a')
    df.to_excel(writer, sheet_name='all', index=False)
    writer.close()


programs = ['A', 'B', 'C', 'D', 'E']
programs_dic = {
    'A': 'python@cpython',
    'B': 'apache@hive',
    'C': 'pypa@pip',
    'D': 'akka@akka',
    'E': 'opf@openproject',
}

if __name__ == "__main__":
    models_dic = {
        'Without': ['Without'],
        'Random': ['skip20', 'skip40', 'skip60', 'skip80', 'skip100'],
        'PCI_5models': ['IHT+RF', 'no_sampling+BB', 'no_sampling+CSP', 'NCR+DT', 'no_sampling+LOF']
    }
    for pro in programs:
        for t in models_dic.keys():
            process_sim_data(pro, t, models_dic[t])
        compare(pro)
        merge_result(pro)
