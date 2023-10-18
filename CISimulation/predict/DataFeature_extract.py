import pandas as pd
import numpy as np
import math
import csv
import os
import traceback
import sys
from utils.commit_util import github_commit
from utils.file_util import readCIData, readCOData
from predict.Config import *

# get all branch and number them (sort them in descending order according to the order of appearance of the branches)
def get_branch(CI_df):
    branch_count = {}
    branch_index = {}
    count = 0
    branch = CI_df['branch']
    for i in branch:
        if i in branch_count:
            branch_count[i] += 1
        else:
            branch_count[i] = 1
    count_sort = sorted(branch_count.items(), key=lambda x: x[1], reverse=True)
    for i in count_sort:
        branch_index[i[0]] = count
        count += 1

    return branch_index

def cal_time_elapse(cur_build_start, prev_build_start):
    if type(prev_build_start) == float and np.isnan(prev_build_start):
        return np.nan
    time_elapse = (cur_build_start - prev_build_start).total_seconds() / (24 * 3600)
    return time_elapse

def previous_build_info(cur_build, branch):
    branch_name = cur_build['branch']
    cur_build_id = cur_build['id']
    cur_build_state = cur_build['state']
    cur_build_start = cur_build['started_at']
    branch_item = [cur_build_id, cur_build_state, cur_build_start]

    prev_build_id = np.nan
    prev_build_state = np.nan
    prev_build_start = np.nan

    prev_failed_build_id = np.nan
    prev_failed_build_start = np.nan
    if branch:
        if branch_name in branch:
            prev_build_id = branch[branch_name][-1][0]
            prev_build_state = branch[branch_name][-1][1]
            prev_build_start = branch[branch_name][-1][2]
            history = branch[branch_name]
            for i in range(len(history)-1, -1, -1):
                prev_build = history[i]
                if prev_build[1] == False:
                    prev_failed_build_id = prev_build[0]
                    prev_failed_build_start = prev_build[2]
                    break
            branch[branch_name].append(branch_item)
        else:
            branch[branch_name] = []
            branch[branch_name].append(branch_item)
    else:
        branch[branch_name] = []
        branch[branch_name].append(branch_item)

    time_elapse = cal_time_elapse(cur_build_start, prev_build_start)
    time_last_failed_build = cal_time_elapse(cur_build_start, prev_failed_build_start)

    return [prev_build_id, prev_failed_build_id, prev_build_state, time_elapse, time_last_failed_build]

# get same_committer to distinguish branches
def whether_same_committer(CI_df, CO_df, cur_build_id, prev_build_id):
    if np.isnan(prev_build_id):
        print("unvalid prev_build_id of " + str(cur_build_id))
        return np.nan
    cur_build = CI_df.loc[CI_df['id']==cur_build_id].iloc[0, :]
    prev_build = CI_df.loc[CI_df['id']==prev_build_id].iloc[0, :]

    flag = np.nan
    try:
        cur_commit = CO_df.loc[CO_df['sha']==cur_build['trigger_commit']].iloc[0, :]
        prev_commit = CO_df.loc[CO_df['sha'] == prev_build['trigger_commit']].iloc[0, :]
        flag = 1 if cur_commit['committer'] == prev_commit['committer'] else 0
    except Exception as e:
        print(e)

    return flag

# get committer_history, committer_recent, and committer_exp to distinguish branches
def committer_info(CO_df, cur_build, committers_histories):
    commit = CO_df.loc[CO_df['sha'] == cur_build['trigger_commit']]
    if len(commit) == 0:
        sys.stderr.write("error in func committer_info, can't find cur_build_trigger_commit in CO_df")
        return [np.nan] * 3
    commit = commit.iloc[0, :]

    branch = cur_build['branch']
    build_state = cur_build['state']
    committer = commit['committer'] + "#" + branch

    build_l = []
    if committers_histories:
        if committer in committers_histories:
            build_l = committers_histories[committer][:]
            committers_histories[committer].append(build_state)
        else:
            committers_histories[committer] = []
            committers_histories[committer].append(build_state)
    else:
        committers_histories[committer] = []
        committers_histories[committer].append(build_state)

    com_history = np.nan
    com_recent = np.nan
    com_exp = np.nan
    if len(build_l) > 0:
        num_passed_build = 0
        for build_status in build_l:
            num_passed_build += build_status
        success_rate = num_passed_build / len(build_l)
        com_history = 1 - success_rate

    if len(build_l) >= 5:
        num_passed_build = 0
        for i in range(-5, 0):
            num_passed_build += build_l[i]
        success_rate = num_passed_build / 5
        com_recent = 1 - success_rate

    com_exp = len(build_l)

    return [commit['committer'], com_history, com_recent, com_exp]

def gaussian_threat(build_list):
    build_index = []
    build_result = []
    i = len(build_list)
    for j in range(0, i - 1):
        if build_list[j] != 1 and build_list[j] != 0:
            continue
        else:
            build_index.append(j)
            build_result.append(build_list[j])

    result = 0.0
    for x in range(0, len(build_index)):
        if build_result[x] == 0:
            ft = len(build_result) - x  # contain this build
            Dt = 0.0
            for y in range(x + 1, len(build_index)):
                Dt = Dt + (build_index[y] - build_index[x])
            dt = Dt / ft
            if ft == 0 or dt == 0:
                result += 0
            else:
                result += (1 / (math.sqrt(2.0 * math.pi) * dt)) * math.exp(
                    0.0 - math.pow(ft, 2) / (math.pow(dt, 2) * 2.0))
        else:
            pass
    return result

def get_project_history_metrics(CI_df, cur_index, cur_branch):
    project_history = np.nan
    project_recent = np.nan

    build_history = CI_df.loc[CI_df['branch'] == cur_branch]
    build_history = build_history.loc[:(cur_index-1)]['state']
    len_build_history = len(build_history)

    if len_build_history > 0:
        num_passed_build = build_history.sum()
        project_history = (len_build_history - num_passed_build) / len_build_history

    if len_build_history >= 5:
        num_passed_build = build_history[-5:].sum()
        project_recent = (5 - num_passed_build) / 5

    gauss_re = np.nan
    if len_build_history > 0:
        gauss_re = gaussian_threat(build_history.tolist())

    return [project_history, project_recent, gauss_re]

def git_diff_infos(owner, repo, commits):
    def judgeFileType(filename):
        src_type = ['.java', '.rb', '.cpp', '.c', '.py', '.php', '.cc', '.cxx']
        conf_type = ['.xml', '.yml']
        file_suffix = os.path.splitext(filename)[-1]
        if file_suffix in src_type:
            return "src"
        elif file_suffix in conf_type:
            return "conf"
        else:
            return "other"

    result = {}
    lines_add = 0
    lines_deleted = 0
    added_files = set()
    removed_files = set()
    modified_files = set()
    msg_length = 0
    no_src = 0
    no_conf = 0

    for sha in commits:
        raw_commit = github_commit(owner, repo, sha)
        try:
            lines_add += raw_commit['stats']['additions']
            lines_deleted += raw_commit['stats']['deletions']

            msg_length += len(raw_commit['commit']['message'])

            for raw_file in raw_commit['files']:
                filename = raw_file['filename']
                if raw_file['status'] == 'added':
                    added_files.add(filename)
                elif raw_file['status'] == 'removed':
                    removed_files.add(filename)
                elif raw_file['status'] == 'modified':
                    modified_files.add(filename)
                fileType = judgeFileType(filename)
                if fileType == "src":
                    no_src = 1
                elif fileType == "conf":
                    no_conf = 1
        except:
            sys.stderr.write(traceback.format_exc())
            continue

    result['lines_add'] = lines_add
    result['lines_deleted'] = lines_deleted
    result['files_added'] = len(added_files)
    result['files_removed'] = len(removed_files)
    result['files_modified'] = len(modified_files)
    result['commit_msg_length'] = msg_length * 1.0 / len(commits)
    result['no_src_edited'] = no_src
    result['no_config_edited'] = no_conf
    result['no_src_config_edited'] = no_src | no_conf

    files_edited = len(added_files) + len(removed_files) + len(modified_files)
    avg_files_edited = files_edited / len(commits)
    commit_msg_length = msg_length * 1.0 / len(commits)
    no_src_config = no_src | no_conf

    return [lines_add, lines_deleted, files_edited, avg_files_edited, commit_msg_length, no_src, no_conf, no_src_config]

def collision_files_developers(owner, repo, commits):
    developers = {}
    files = {}
    for sha in commits:
        raw_commit = github_commit(owner, repo, sha)
        try:
            c = raw_commit['commit']['committer']['email']
            if c in developers.keys():
                developers[c] += 1
            else:
                developers[c] = 1
            for raw_file in raw_commit['files']:
                filename = raw_file['filename']
                if filename in files:
                    files[filename] += 1
                else:
                    files[filename] = 1
        except:
            sys.stderr.write(traceback.format_exc())

    collision_files = 0
    collision_developers = 0
    developer_count = len(developers.keys())
    for k,v in files.items():
        if v > 1:
            collision_files += 1
    for k,v in developers.items():
        if v > 1:
            collision_developers += 1

    return [collision_files, collision_developers, developer_count]

def feature_extract(pro):
    owner = pro.split('@')[0]
    repo = pro.split('@')[1]
    fileCI = f"{PROJECT_DIR}/{pro}/repo-data-travis.csv"
    fileCO = f"{PROJECT_DIR}/{pro}/repo-data-commits.csv"
    outpath = f"{FEATURE_DIR}/{pro}_feature.csv"

    CI_df = readCIData(fileCI)
    CO_df = readCOData(fileCO)

    branch = {}
    committer = {}

    outfile = open(outpath, 'w', encoding='utf-8', newline='')
    writer = csv.writer(outfile)
    title = ['build_id', 'build_result', 'branch', 'commit_last_build', 'day_time', 'weekday', 'month_day',
             'additions_last_build', 'deletion_last_build', 'total_files_pushed', 'avg_file_committed',
             'commit_msg_length', 'no_src_edited', 'no_config_edited', 'no_src_config_edited',
             'last_build_result', 'time_elapse', 'time_last_failed_build',
             'committer', 'committer_history', 'committer_recent', 'committer_exp', 'same_committer',
             'collision_files', 'collision_developers', 'developer_count',
             'project_history', 'project_recent', 'gaussian_threat']
    writer.writerow(title)

    total_row = CI_df.shape[0]
    for index, row in CI_df.iterrows():
        print('%s/%s' %(index+1, total_row))
        result = []

        cur_build_id = row['id']
        cur_build_state = row['state']
        all_commits = row['all_commits'].split(';')
        cur_branch = row['branch']
        result.append(cur_build_id)
        result.append(cur_build_state)
        result.append(cur_branch)
        result.append(len(all_commits))

        build_time = row['started_at']
        day_time = build_time.hour
        day_week = build_time.dayofweek
        day_month = build_time.day
        result.append(day_time)
        result.append(day_week)
        result.append(day_month)

        code_change_metrics = git_diff_infos(owner, repo, all_commits)
        result.extend(code_change_metrics)

        prev_build_metrics = previous_build_info(row, branch)
        prev_build_id = prev_build_metrics[0]
        prev_failed_build_id = prev_build_metrics[1]
        result.extend(prev_build_metrics[2:])

        committer_metrics = committer_info(CO_df, row, committer)
        result.extend(committer_metrics)

        same_committer = whether_same_committer(CI_df, CO_df, cur_build_id, prev_build_id)
        result.append(same_committer)

        collision_metrics = collision_files_developers(owner, repo, all_commits)
        result.extend(collision_metrics)

        project_metrics = get_project_history_metrics(CI_df, index, cur_branch)
        result.extend(project_metrics)

        writer.writerow(result)


if __name__ == '__main__':
    pros = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']
    for p in pros:
        print(p + " begin")
        feature_extract(p)
