## 文件目录结构说明

- repo-data-travis-fixTime_total.xlsx 根据"缺陷修复时间"的计算规则，统计了所有项目中的缺陷修复信息。该文件中的每条记录的字段说明：prev_build_id、cur_build_id 分别代表“缺陷引入”和“缺陷修复”的build_id（每个build可关联到一个或多个commit），prev_started_at和cur_finished_at 分别代表“cur_build”的开始时间和“prev_build”的完成时间，这两个时间的差值就是“缺陷修复”的时间, 即fixTime(minute)，fixType是缺陷修复的类型（包含两种：立即修复、延迟修复），具体分类依据见readme中对 FixTime.py 脚本的介绍
- repo-data-travis_process.csv 按年份统计了总构建数量超过4000的项目，每一年的构建触发类型数量（Push、PR、cron、API），根据脚本 sortByYear.py 得到
- repo-data-travis_process_2020_2021.csv 对repo-data-travis_process.csv进一步加工，统计了2020~2021两年，项目的构建总数、四种构建触发类型的数量。基于该文件选择了TOP1/2/7/8/14/15六个项目作为实验项目（先筛选2020~2021两年的构建总数超过4000的项目，然后选择前2，中2，后2的六个项目），根据脚本 sortByYear.py 得到
- 六个 owner@repo 文件夹就是针对挑选出的6个项目，进一步爬取数据。以opf@openproject项目为例：
  - commits/ : 存放该项目的历史commit记录，每个commit每一个json文件，通过github api爬取得到
  - repo-data-commits.csv：对 commits/文件中所有commit记录的汇总，提取关键字的（sha、author、committer、addition、deletion、file_changed），根据脚本GHTorrentTest.py得到
  - repo-data-travis.json：借助 travis api爬取的该项目的历史build记录
  - repo-data-travis.csv：对 repo-data-travis.json 的进一步加工，添加字段（git_all_commits），根据脚本 GHTorrentTest.py得到
  - repo-data-travis-process.csv：对 repo-data-travis.csv 的进一步加工，添加字段（file_changed，该次build包含的所有commit的修改过的文件集合），根据脚本 FixTime.py 得到
  - repo-data-travis-fixTime.csv：对 repo-data-travis-process.csv 的进一步加工，计算所有缺陷build的修复时间（区分分支），根据脚本 FixTime.py 得到
  - repo-data-travis-fixTime_process.csv：对 repo-data-travis-fixTime.csv 的进一步加工，从中筛选出符合需求的记录（需满足“缺陷引入”和“缺陷修复”在同一天，即两次build的触发时间和完成时间在同一天）
  - {project}_feature.csv：从每个项目中提取到的特征，根据 DataFeature_extract.py 脚本得到
  - {project}_feature_process：对特征进行加工，比如：处理某些字段为空值的行



## 脚本说明

- ScripyTest.py：统计gh-active-projects.csv中每个项目的构建总数，结果保存为 res_top5000.csv
- GHTorrentTest.py：下载项目的build数据、commit数据（源数据保存为json文件，加工后保存为csv文件）
- sortByYear.py：统计2020~2021年项目的build数据（按触发类型进行了分类统计）
- FixTime.py：统计项目中的含缺陷构建的fixTime（区分分支）
- DataFeature_extract.py：对每个项目进行特征提取
- Calibration.py：获取仿真校准数据（仿真模型中所需的配置项）

### 缺陷修复时间的计算说明

- 回溯。将文件变动视作缺陷的引入和修复活动，从一个pass build开始向前回溯到最近的一个fail build，pass build 变动的文件集合记作 cur_files，fail build 变动的文件集合记作 prev_files，prev_files 具体可分为两个部分：trigger commit 变动的文件集合，记作 trigger_files；other commits（fail build 中除 trigger commit 外的其他 commits 集合）变动的文件集合，记作 other_files。基于：cur_files，prev_files，trigger_files，other_files 对缺陷修复类型进行判断：

  - 如果 cur_files 是 prev_files 的子集，则进一步判断
    - 如果 cur_files 是 trigger_files 的子集，则判定为"立即修复"，修复时间 = cur_build_finished_at 减去 trigger_commit_date
    - 否则，判定为"延迟修复"，修复时间 = cur_build_finished_at 减去 other commits 中提交时间最早的 commit
  - 否则，继续向前回溯
  - ==如果图片看不了：请看readme.pdf==

  ![1651108720167](C:\Users\endeavor\AppData\Roaming\Typora\typora-user-images\1651108720167.png)



## 程序使用须知

### 使用须知

GHTorrentTest.py 是爬虫脚本。*代表 owner_repo，运行条件：

- ==运行脚本前，需要补充 github_token 的值，如果提示 github_token 无效，可能是当前程序中的 github_token 过期了，需要重新生成一个==。
- 因为github api对每个用户限制是每小时最多爬取5000条数据，限制范围是每个用户的token
- github_token的获取教程：https://www.cnblogs.com/chenyablog/p/15397548.html
