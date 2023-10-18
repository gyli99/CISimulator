import pandas as pd
import numpy as np
from utils.file_util import readCIData, readCOData
from calibration.Config import *

# calculate commitInterval_mean
def commitInterval_mean(pro, df_CO):
    dateInfos = df_CO['date'].tolist()
    intervals = []
    for i in range(1, len(dateInfos)):
        if(dateInfos[i].date().__eq__(dateInfos[i-1].date())):
            # only if the two commits are submitted in the same day, the time interval between them would be calculated
            diff = (dateInfos[i] - dateInfos[i-1]).total_seconds() / 60
            if diff <= 480:
                # since we adopted 8-hour work daily in our paper,
                # only commits within eight hours are considered to be submitted on the same day.
                intervals.append(diff)
    # remove the first and last 5%
    intervals = np.sort(intervals)
    num = int(len(intervals) * 0.05)
    intervals = intervals[num: (len(intervals) - num)]

    inter_mean = np.mean(intervals)
    inter_xia = np.quantile(intervals, 0.25, interpolation='lower')
    inter_mid = np.quantile(intervals, 0.5, interpolation='lower')
    print('commitInterval_mean(mins): %.2f' % inter_mean)
    print('commitInterval_mean(xia): %.2f' % inter_xia)
    print('commitInterval_mean(mid): %.2f' % inter_mid)
    return [inter_mean, inter_mid, inter_xia]


# calculate push、PR、skipPush、skipPR
def commitType(df_CI):
    df_CI["duration"] = df_CI[["finished_at", "started_at"]].apply(lambda x: (x["finished_at"] - x["started_at"]).total_seconds() / 60, axis=1)

    pushCommit = df_CI.loc[df_CI['event_type']=='push']
    prCommit = df_CI.loc[df_CI['event_type']=='pull_request']

    # calculate the rate of push and PR
    pushRate = len(pushCommit) / (len(pushCommit) + len(prCommit))
    prRate = 1 - pushRate

    # calculate the probability of skipPR and skipPush
    skipPush = (pushCommit['num_all_commits'].sum() - len(pushCommit)) / pushCommit['num_all_commits'].sum()
    skipPR = (prCommit['num_all_commits'].sum() - len(prCommit)) / prCommit['num_all_commits'].sum()

    print('prRate: %.2f' % prRate)
    print('skipPR: %.2f' % skipPR)
    print('skipPush: %.2f' % skipPush)
    return [prRate, skipPR, skipPush]


# calculate the fail rate
def resultType(df_CI):
    failCommit = df_CI.loc[df_CI['state']==0]
    failRate = len(failCommit) / len(df_CI)
    print('failRate: %.2f' % failRate)
    return failRate

# calculate tha rate of PushPassDelay, PushFailDelay, PRPassDelay, and PRFailDelay
def eventBuildTime(df_CI):
    def helper(eventType, buildResult):
        builds = df_CI.loc[(df_CI['event_type'] == eventType) & (df_CI['state'] == buildResult)]
        durations = builds['duration'].tolist()
        durations = np.sort(durations)
        num = int(len(durations) * 0.05)
        durations = durations[num: (len(durations) - num)]

        buildDelay_mean = np.mean(durations)
        buildDelay_std = np.std(durations)
        return buildDelay_mean, buildDelay_std
    PushPassDelay_mean, PushPassDelay_std = helper('push', 1)
    PushFailDelay_mean, PushFailDelay_std = helper('push', 0)
    PRPassDelay_mean, PRPassDelay_std = helper('pull_request', 1)
    PRFailDelay_mean, PRFailDelay_std = helper('pull_request', 0)
    print('PRPassDelay, mean: %.2f, std: %.2f' % (PRPassDelay_mean, PRPassDelay_std))
    print('PRFailDelay, mean: %.2f, std: %.2f' % (PRFailDelay_mean, PRFailDelay_std))
    print('PushPassDelay, mean: %.2f, std: %.2f' % (PushPassDelay_mean, PushPassDelay_std))
    print('PushFailDelay, mean: %.2f, std: %.2f' % (PushFailDelay_mean, PushFailDelay_std))
    return [PRPassDelay_mean, PRPassDelay_std, PRFailDelay_mean, PRFailDelay_std,
            PushPassDelay_mean, PushPassDelay_std, PushFailDelay_mean, PushFailDelay_std]


def CIBuildTime(df_CI):
    def helper(buildResult):
        builds = df_CI.loc[df_CI['state'] == buildResult]
        durations = builds['duration'].tolist()
        durations = np.sort(durations)
        num = int(len(durations) * 0.05)
        durations = durations[num: (len(durations) - num)]
        buildDelay_mean = np.mean(durations)
        buildDelay_std = np.std(durations)
        return buildDelay_mean, buildDelay_std
    PassDelay_mean, PassDelay_std = helper(1)
    FailDelay_mean, FailDelay_std = helper(0)
    print('PassDelay, mean: %.2f, std: %.2f' % (PassDelay_mean, PassDelay_std))
    print('FailDelay, mean: %.2f, std: %.2f' % (FailDelay_mean, FailDelay_std))
    return [PassDelay_mean, PassDelay_std, FailDelay_mean, FailDelay_std]

def branchRate(df_CI):
    total = df_CI.shape[0]
    grouped = df_CI.groupby('branch')
    groups = dict()
    for name, group in grouped:
        groups[name] = len(group)*1.0 / total
    sort_groups = sorted(groups.items(), key=lambda d: d[1], reverse=True)
    sort_groups = dict(sort_groups)
    print('total builds: %d, total_branch: %d' % (total, len(groups)))
    for key in sort_groups.keys():
        print("branch: %s, s: %f" % (key, groups.get(key)))
    master = 0
    default_branches = ['master', 'main', 'dev', 'develop']
    for branch in default_branches:
        master = groups.get(branch, 0)
        if master > 0:
            break
    return master

def commit_seq(l):
    pp, pf, ff, fp = 0, 0, 0, 0
    for i in range(0, len(l) - 1):
        if l[i] == 1 and l[i + 1] == 1:
            pp += 1
        elif l[i] == 1 and l[i + 1] == 0:
            pf += 1
        elif l[i] == 0 and l[i + 1] == 0:
            ff += 1
        elif l[i] == 0 and l[i + 1] == 1:
            fp += 1

    pp_rate = pp / (pp + pf) if pp + pf != 0 else np.nan
    ff_rate = ff / (ff + fp) if ff + fp != 0 else np.nan
    return pp_rate, ff_rate

def splitCIData(df_CI):
    split_num = 8650
    total_num = df_CI.shape[0]
    start = 0
    end = start + split_num
    split_data = df_CI.iloc[start: end, :]
    return f"{start}-{end}", split_data

def splitCOData(df_CI, df_CO):
    commit_list = []
    for index, row in df_CI.iterrows():
        l = row['all_commits'].split(";")
        commit_list.extend(l)
    split_data = df_CO.loc[df_CO['sha'].isin(commit_list)]
    return split_data


if __name__ == '__main__':
    pros = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']
    columns = ['split_index', 'CI_num', 'commit_num', 'inter_mean', 'inter_mid', 'inter_xia',
               'prRate', 'skipPR', 'skipPush', 'failRate', 'PRPassDelay_mean', 'PRPassDelay_std',
               'PRFailDelay_mean', 'PRFailDelay_std', 'PushPassDelay_mean', 'PushPassDelay_std',
               'PushFailDelay_mean', 'PushFailDelay_std', 'master', 'c_pp', 'c_ff']
    output_path = f"{OUTPUT_DIR}/calibration_split_data_f.xlsx"
    writer = pd.ExcelWriter(output_path, mode='w')

    for pro in pros:
        fileCI = f"{PROJECT_DIR}/{pro}/repo-data-travis.csv"
        fileCO = f"{PROJECT_DIR}/{pro}/repo-data-commits.csv"
        df_CI = readCIData(fileCI)
        df_CO = readCOData(fileCO)
        print(df_CO.iloc[0]['date'])
        datas = []
        for i in range(1):
            split_index, df_CI_split = splitCIData(df_CI)
            df_CO_split = splitCOData(df_CI_split, df_CO)
            print(pro)
            data = [split_index, df_CI_split.shape[0], df_CO_split.shape[0]]
            data.extend(commitInterval_mean(pro, df_CO_split))
            data.extend(commitType(df_CI_split))
            data.append(resultType(df_CI_split))
            data.extend(eventBuildTime(df_CI_split))
            data.append(branchRate(df_CI_split))
            data.extend(commit_seq(df_CI_split['state'].tolist()))
            datas.append(data)
        df_file = pd.DataFrame(data=datas, columns=columns)
