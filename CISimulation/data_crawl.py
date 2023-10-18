import requests
import json
import os
import pandas as pd
from datetime import datetime
from utils.commit_util import github_commit

def travis_builds_json(owner, repo, type):
    dir_path = "projects/" + owner + "@" + repo
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filepath = "projects/" + owner + '@' + repo + '/repo-data-travis.json'
    if os.path.exists(filepath):
        print(filepath + " exists!")
        return

    url = "https://api.travis-ci." + type + "/v3/repo/" + owner + "%2F" + repo + '/builds?limit=100'
    builds_info = requests.get(url).json()
    total_builds_info = builds_info
    while builds_info['@pagination']['is_last'] == False:
        print('read next')
        url = "https://api.travis-ci." + type + builds_info['@pagination']['next']['@href']
        builds_info = requests.get(url).json()
        total_builds_info['builds'].extend(builds_info['builds'])

    with open(filepath, 'w') as fp:
        json.dump(total_builds_info, fp)
        print(filepath + " write to local json file, success!")

def build_all_commits(owner, repo, build_id):
    filepath = "projects/" + owner + '@' + repo + '/repo-data-travis.json'

    commits = []
    prev_build = None
    with open(filepath, 'r') as fp:
        total_builds_info = json.load(fp)['builds']
        builds_map = {}
        commits_map = {}
        for build in total_builds_info:
            bid = str(build['id'])
            cid = str(build['commit']['sha'])
            builds_map[bid] = build
            commits_map[cid] = bid
        if build_id not in builds_map.keys():
            print("projects/" + owner + '@' + repo + ' dont exist ' + build_id)
            return commits

        build = builds_map.get(build_id)
        commits.append(build['commit']['sha'])
        prev_commit_resolution_status = "no_previous_build"
        try:
            cur_commit = github_commit(owner, repo, build['commit']['sha'])
            while True:
                if cur_commit is None:
                    print('invalid commit' + cur_commit)
                    break
                elif len(cur_commit['parents']) > 1:
                    print('merge_found ' + cur_commit['sha'])
                    prev_commit_resolution_status = "merge_found"
                    break
                elif len(cur_commit['parents']) == 1:
                    prev_commit = cur_commit['parents'][0]['sha']
                    if prev_commit in commits_map:
                        print('prev build commit ' + prev_commit + ", build_id:" + commits_map[prev_commit])
                        prev_build = commits_map[prev_commit]
                        prev_commit_resolution_status = "build_found"
                        break
                    elif cur_commit['sha'] != build['commit']['sha']:
                        commits.append(cur_commit['sha'])
                    cur_commit = github_commit(owner, repo, prev_commit)
                else:
                    print('parent commits of %s is None' % cur_commit)
                    break
        except Exception as e:
            print(e)
        all_commits = ';'.join(commits)
        num_all_commits = len(commits)
        return all_commits, num_all_commits, prev_build, prev_commit_resolution_status

def travis_build_jsonToCsv(owner, repo):
    def getAllJobs(jobs_json):
        jobs = []
        for j in jobs_json:
            jobs.append(j['id'])
        return jobs

    def judgeIs2020_2021(start_time):
        if start_time is None:
            return False
        year = start_time.split("-")[0]
        if year == '2020' or year == '2021':
            return True
        return False

    result = []

    filepath = "projects/" + owner + '@' + repo + '/repo-data-travis.json'
    with open(filepath, 'r') as fp:
        total_builds_info = json.load(fp)['builds']
        index = 1
        for build in total_builds_info:
            print(index)
            index += 1
            id = str(build['id'])
            number = build['number']
            state = build['state']
            event_type = build['event_type']
            previous_state = build['previous_state']
            build_duration = build['duration']
            started_at = build['started_at']
            finished_at = build['finished_at']

            if judgeIs2020_2021(started_at) == False:
                continue

            branch = build['branch']['name']
            trigger_commit = build['commit']['sha']
            all_commits, num_all_commits, previous_build, prev_commit_resolution_status = build_all_commits(owner, repo, id)
            all_jobs = getAllJobs(build['jobs'])
            num_all_jobs = len(all_jobs)

            build_info = [id, number, state, event_type, previous_build, previous_state, prev_commit_resolution_status, build_duration, started_at, finished_at, branch, trigger_commit, all_commits, num_all_commits, all_jobs, num_all_jobs]
            result.append(build_info)

    output_file = "projects/" + owner + '@' + repo + '/repo-data-travis.csv'
    columns = ['id', 'number', 'state', 'event_type', 'previous_build', 'previous_state', 'prev_commit_resolution_status', 'build_duration', 'started_at', 'finished_at', 'branch', 'trigger_commit', 'all_commits', 'num_all_commits', 'all_jobs', 'num_all_jobs']
    df = pd.DataFrame(data=result, columns=columns)
    df.to_csv(output_file, index=False, header=True)

def commits_jsonToCsv(owner, repo):
    dir_path = "projects/" + owner + "@" + repo + "/commits"
    output = "projects/" + owner + "@" + repo + "/repo-data-commits.csv"
    if os.path.exists(output):
        print(output + " existed")
        return
    files = os.listdir(dir_path)
    total_commits_info = []
    for f in files:
        fpath = dir_path + "/" + f
        with open(fpath, 'r') as fp:
            raw_commit = json.load(fp)
            try:
                sha = raw_commit['sha']
                author = raw_commit['commit']['author']['email']
                committer = raw_commit['commit']['committer']['email']
                date = raw_commit['commit']['committer']['date']
                message_length = len(raw_commit['commit']['message'])
                parents = ';'.join([p['sha'] for p in raw_commit['parents']])
                additions = raw_commit['stats']['additions']
                deletions = raw_commit['stats']['deletions']
                files_changed = ';'.join([f['filename'] for f in raw_commit['files']])
                num_files_changed = len(files_changed.split(';'))
                commit_info = [sha, author, committer, date, message_length, parents, additions, deletions, files_changed, num_files_changed]
                total_commits_info.append(commit_info)
            except Exception as e:
                print("%s, %s" % (e, fpath))
    columns = ['sha', 'author', 'committer', 'date', 'message_length', 'parents', 'additions', 'deletions', 'files_changed', 'num_files_changed']
    df = pd.DataFrame(data=total_commits_info, columns=columns)
    df.to_csv(output, index=False, header=True)

def countOf2021(owner, repo):
    filepath = "projects/" + owner + '@' + repo + '/repo-data-travis.json'
    with open(filepath, 'r') as fp:
        total_builds_info = json.load(fp)['builds']
        count = 0
        pass_builds = 0
        fail_builds = 0
        dic_type = {}
        for build in total_builds_info:
            id = str(build['id'])
            state = build['state']
            event_type = build['event_type']
            started_at = build['started_at']
            finished_at = build['finished_at']

            started_at = datetime(started_at)
            finished_at = datetime(finished_at)

            if started_at.year != 2021 and finished_at.year != 2021:
                continue

            if state == 'passed':
                pass_builds += 1
            elif state == 'failed':
                fail_builds += 1
            if event_type not in dic_type.keys():
                dic_type[event_type] = 0
            dic_type[event_type] += 1


if __name__ == '__main__':
    pros = ['python@cpython', 'frappe@erpnext', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']
    for p in ['pypa@warehouse']:
        print(p + " begin")
        owner = p.split('@')[0]
        repo = p.split('@')[1]
        travis_builds_json(owner, repo, "com")
        travis_build_jsonToCsv(owner, repo)
        commits_jsonToCsv(owner, repo)