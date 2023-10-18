import os
import json
import requests
import time

github_token = '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'

# get commit info
# if the commit does not exist, download from github and save it.
# if it exists, read directly from the file
def github_commit(owner, repo, sha):
    dir_path = "../projects/" + owner + "@" + repo + "/commits"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filepath = "../projects/" +owner + '@' + repo + '/commits/' + sha + '.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as fp:
            # print(filepath + ' exist!')
            json_data = json.load(fp)
            return json_data

    url = "https://api.github.com/repos/%s/%s/commits/%s" % (owner, repo, sha)
    headers = {"Authorization": "token " + github_token}
    try:
        r = requests.get(url, headers=headers)
        reset = r.headers['X-RateLimit-Reset']
        remain = r.headers['X-RateLimit-Remaining']
        if int(remain) < 10:
            to_sleep = int(reset) - int(time.time()) + 2
            print("Request limit reached, sleeping for %s secs" % to_sleep)
            print("recover time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(reset))))
            time.sleep(to_sleep)

        response = r.json()
        with open(filepath, 'w') as fp:
            json.dump(response, fp)
            print(filepath + ' dont exist. request from Github and write to local json file, success!')
            return response
    except Exception as e:
        print(e)
        return None
