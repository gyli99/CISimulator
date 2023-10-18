import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.file_util import readCIData, readCOData
from calibration.Config import *
from scipy.stats import kstest
import seaborn as sns

# calculate commitInterval_mean
def commitInterval_mean(pro, df_CO):
    dateInfos = df_CO['date'].tolist()
    writer = pd.ExcelWriter('./temp/dataInfos.xlsx')
    df = pd.DataFrame(dateInfos)
    df.to_excel(writer, index=False)
    writer.close()
    intervals = []
    for i in range(1, len(dateInfos)):
        # only if the two commits are submitted in the same day, the time interval between them would be calculated
        if(dateInfos[i].date().__eq__(dateInfos[i-1].date())):
            diff = (dateInfos[i] - dateInfos[i-1]).total_seconds() / 60
            # since we adopted 8-hour work daily in our paper,
            # only commits within eight hours are considered to be submitted on the same day.
            if diff <= 480:
                intervals.append(diff)

    # remove the first and last 5%
    intervals = np.sort(intervals)
    num = int(len(intervals) * 0.05)
    intervals = intervals[num: (len(intervals) - num)]

    inter_mean = np.mean(intervals)
    inter_xia = np.quantile(intervals, 0.25, interpolation='lower')
    inter_mid = np.quantile(intervals, 0.5, interpolation='lower')
    inter_std = np.std(intervals)
    inter_var = np.var(intervals)
    print(kstest(intervals, 'expon'))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    p1 = sns.kdeplot(intervals)
    print()
    print('inter_mean: %.2f' % inter_mean)
    print('inter_var: %.2f' % inter_var)
    return intervals

# calculate push、PR、skipPush、skipPR
def commitType(df_CI):
    df_CI["duration"] = df_CI[["finished_at", "started_at"]].apply(
        lambda x: (x["finished_at"] - x["started_at"]).total_seconds() / 60, axis=1
    )

    pushCommit = df_CI.loc[df_CI['event_type']=='push']
    prCommit = df_CI.loc[df_CI['event_type']=='pull_request']

    pushRate = len(pushCommit) / (len(pushCommit) + len(prCommit))
    prRate = 1 - pushRate

    skipPush = (pushCommit['num_all_commits'].sum() - len(pushCommit)) / pushCommit['num_all_commits'].sum()
    skipPR = (prCommit['num_all_commits'].sum() - len(prCommit)) / prCommit['num_all_commits'].sum()

    print('prRate: %.2f' % prRate)
    print('pushRate: %.2f' % pushRate)
    print('skipPR: %.2f' % skipPR)
    print('skipPush: %.2f' % skipPush)


# calculate fail
def resultType(df_CI):
    failCommit = df_CI.loc[df_CI['state']==0]
    failRate = len(failCommit) / len(df_CI)
    print('failRate: %.2f' % failRate)

# calculate PushPassDelay, PushFailDelay, PRPassDelay, PRFailDelay
def eventBuildTime(df_CI):
    def helper(eventType, buildResult):
        builds = df_CI.loc[(df_CI['event_type'] == eventType) & (df_CI['state'] == buildResult)]
        durations = builds['duration'].tolist()
        durations = np.sort(durations)
        num = int(len(durations) * 0.05)
        durations = durations[num: (len(durations) - num)]

        buildDelay_mean = np.mean(durations)
        buildDelay_std = np.std(durations)
        return durations, buildDelay_mean, buildDelay_std

    def helper_cauchy(eventType, buildResult):
        builds = df_CI.loc[(df_CI['event_type'] == eventType) & (df_CI['state'] == buildResult)]
        durations = builds['duration'].tolist()
        durations = np.sort(durations)
        num = int(len(durations) * 0.05)
        durations = durations[num: (len(durations) - num)]
        buildDelay_loc = np.median(durations)
        buildDelay_scale = (np.percentile(durations, 75) - np.percentile(durations, 25)) / 2
        return durations, buildDelay_loc, buildDelay_scale

    def helper_laplace(eventType, buildResult):
        builds = df_CI.loc[(df_CI['event_type'] == eventType) & (df_CI['state'] == buildResult)]
        durations = builds['duration'].tolist()
        durations = np.sort(durations)
        num = int(len(durations) * 0.05)
        durations = durations[num: (len(durations) - num)]
        buildDelay_loc = np.mean(durations)
        buildDelay_scale = np.mean(np.abs(durations - buildDelay_loc)) / np.log(2)
        return durations, buildDelay_loc, buildDelay_scale

    PushPassDelays, PushPassDelay_mean, PushPassDelay_std = helper('push', 1)
    PushFailDelays, PushFailDelay_loc, PushFailDelay_scale = helper('push', 0)
    PRPassDelays, PRPassDelay_loc, PRPassDelay_scale = helper('pull_request', 1)
    PRFailDelays, PRFailDelay_mean, PRFailDelay_std = helper('pull_request', 0)
    print('PRPassDelay, loc: %.2f, scale: %.2f' % (PRPassDelay_loc, PRPassDelay_scale))
    print('PRFailDelay, mean: %.2f, std: %.2f' % (PRFailDelay_mean, PRFailDelay_std))
    print('PushPassDelay, mean: %.2f, std: %.2f' % (PushPassDelay_mean, PushPassDelay_std))
    print('PushFailDelay, loc: %.2f, scale: %.2f' % (PushFailDelay_loc, PushFailDelay_scale))
    return PushPassDelays, PushFailDelays, PRPassDelays, PRFailDelays

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

    branchStr = ",".join(map(lambda x: '%.6f' % x, sort_groups.values()))
    print(branchStr)

def splitDataset(df_feature):
    rate = 0.8
    num = int(len(df_feature) * rate)
    train = df_feature.iloc[0:num, :]
    test = df_feature.iloc[num:, :]
    return test

def ks_test(data, mean, std):
    exp = np.random.exponential(mean, size=(len(data)))
    print(f'exponential: {kstest(data, exp)}')

    print(f"normal: {kstest(data, 'norm', args=(mean, std))}")


if __name__ == '__main__':
    pros = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']
    for pro in pros:
        print(f'========== {pro} ==========')
        fileCI = f"{PROJECT_DIR}/{pro}/repo-data-travis.csv"
        fileCO = f"{PROJECT_DIR}/{pro}/repo-data-commits.csv"
        fileFeature = f"{FEATURE_DIR}/{pro}_feature.csv"
        df_CI = readCIData(fileCI)
        df_CO = readCOData(fileCO)
        df_feature = pd.read_csv(fileFeature)
        df_feature = df_feature[['build_id']]

        df_CI = pd.merge(df_CI, df_feature, left_on='id', right_on='build_id', how='inner')

        df_CI = splitDataset(df_CI)
        df_CO = splitDataset(df_CO)

        intervals = commitInterval_mean(pro, df_CO)
        commitType(df_CI)
        resultType(df_CI)
        PushPassDelays, PushFailDelays, PRPassDelays, PRFailDelays = eventBuildTime(df_CI)
        columns = ['Intervals', 'PushPassDelays', 'PushFailDelays', 'PRPassDelays', 'PRFailDelays']
        columns_list = [intervals.tolist(), PushPassDelays.tolist(), PushFailDelays.tolist(),
                        PRPassDelays.tolist(), PRFailDelays.tolist()]

        CIBuildTime(df_CI)
        branchRate(df_CI)
        print()
