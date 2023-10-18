import numpy as np
import pandas as pd
from utils.file_util import readCIData, readCOData

def get_split_commit(pro):
    columns = ['split_index', 'CI_num', 'commit_num', 'inter_mean', 'inter_mid', 'inter_xia',
               'prRate', 'skipPR', 'skipPush', 'failRate', 'PRPassDelay_mean', 'PRPassDelay_std',
               'PRFailDelay_mean', 'PRFailDelay_std', 'PushPassDelay_mean', 'PushPassDelay_std',
               'PushFailDelay_mean', 'PushFailDelay_std', 'master', 'c_pp', 'c_ff', 'end_idx',
               'sampler', 'classifier', 'fail_ff', 'fail_pp', 'pass_ff', 'pass_pp',
               'inter_mean2', 'inter_mid2', 'inter_xia2', 'inter_var',
               'LogPRPassDelay_mean', 'LogPRPassDelay_std',
               'LogPRFailDelay_mean', 'LogPRFailDelay_std',
               'LogPushPassDelay_mean', 'LogPushPassDelay_std',
               'LogPushFailDelay_mean', 'LogPushFailDelay_std']
    filepath = f'/Users/thunder/Documents/Projects/CI-Simulation/model_experiment/simdata/calibration_result.xlsx'
    df_calibration = pd.read_excel(filepath, sheet_name=pro)
    ci_file = f'../projects/{pro}/repo-data-travis.csv'
    df_split = readCIData(ci_file)
    commit_file = f'../projects/{pro}/repo-data-commits.csv'
    df_co = readCOData(commit_file)
    result = []
    for i in range(df_calibration.__len__()):
        row_c = df_calibration.iloc[i]
        start_idx, end_idx = row_c['split_index'].split('-')
        start_idx, end_idx = int(start_idx), int(end_idx)
        df_ci = df_split.iloc[start_idx:end_idx, :]
        df_ci["duration"] = df_ci[["finished_at", "started_at"]].apply(
            lambda x: (x["finished_at"] - x["started_at"]).total_seconds() / 60, axis=1)
        commit_list = []
        for index, row in df_ci.iterrows():
            l = row['all_commits'].split(";")
            commit_list.extend(l)
        split_data = df_co.loc[df_co['sha'].isin(commit_list)]
        inter_list = commitInterval_mean(split_data)
        temp = row_c.tolist()
        if i == 0:
            print(temp.__len__())
        for j in range(len(inter_list)):
            temp.append(inter_list[j])
        delay_list = eventBuildTime(df_ci)
        for j in range(len(delay_list)):
            temp.append(delay_list[j])
        result.append(temp)
    writer = pd.ExcelWriter('./new_model/new_result.xlsx', mode='a')
    df_result = pd.DataFrame(data=result, columns=columns)
    df_result = df_result.fillna(0)
    df_result.to_excel(writer, sheet_name=pro, index=False)
    writer.close()

def commitInterval_mean(df_CO):
    dateInfos = df_CO['date'].tolist()
    intervals = []
    for i in range(1, len(dateInfos)):
        if (dateInfos[i].date().__eq__(dateInfos[i - 1].date())):
            diff = (dateInfos[i] - dateInfos[i - 1]).total_seconds() / 60
            if diff <= 480:
                intervals.append(diff)

    intervals = np.sort(intervals)
    num = int(len(intervals) * 0.05)
    intervals = intervals[num: (len(intervals) - num)]

    inter_mean = np.mean(intervals)
    inter_xia = np.quantile(intervals, 0.25, interpolation='lower')
    inter_mid = np.quantile(intervals, 0.5, interpolation='lower')
    inter_var = np.var(intervals)
    return [inter_mean, inter_mid, inter_xia, inter_var]

def eventBuildTime(df_CI):
    def helper(eventType, buildResult):
        builds = df_CI.loc[(df_CI['event_type'] == eventType) & (df_CI['state'] == buildResult)]
        durations = builds['duration'].tolist()

        durations = np.sort(durations)
        num = int(len(durations) * 0.05)
        durations = durations[num: (len(durations) - num)]

        buildDelay_mean = np.mean(np.log(durations))
        buildDelay_std = np.mean(np.log(durations))
        return buildDelay_mean, buildDelay_std
    PushPassDelay_mean, PushPassDelay_std = helper('push', 1)
    PushFailDelay_mean, PushFailDelay_std = helper('push', 0)
    PRPassDelay_mean, PRPassDelay_std = helper('pull_request', 1)
    PRFailDelay_mean, PRFailDelay_std = helper('pull_request', 0)
    return [PRPassDelay_mean, PRPassDelay_std, PRFailDelay_mean, PRFailDelay_std,
            PushPassDelay_mean, PushPassDelay_std, PushFailDelay_mean, PushFailDelay_std]


if __name__ == "__main__":
    projects = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']
    for p in projects:
        get_split_commit(p)
