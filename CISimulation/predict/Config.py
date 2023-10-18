from utils.file_util import create_dir_if_not_exist

PROJECT_DIR = "../projects"
# FEATURE_DIR = "feature"
FEATURE_DIR = "new_feature"
FEATURE_PROCESS_DIR = "feature_process"
# OUTPUT_DIR = "output"
OUTPUT_DIR = "new_output"


create_dir_if_not_exist(FEATURE_DIR)
create_dir_if_not_exist(FEATURE_PROCESS_DIR)
create_dir_if_not_exist(OUTPUT_DIR)
