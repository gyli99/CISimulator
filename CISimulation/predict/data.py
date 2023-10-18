from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd

# normalization
def feature_normalize1(data):
    ss = StandardScaler()
    ss.fit(data)
    data = ss.transform(data)
    return data

# standardization
def feature_normalize2(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data

# process features, e.g., fill lines with empty values
def feature_process(df):
    # drop three columns：branch、committer、no_src_config_edited
    # empty values：
    # - drop：last_build_result、time_elapse、time_last_failed_build、same_committer
    # - fill with 0：committer_history、committer_recent、committer_exp、project_history、project_recent、gaussian_threat
    df.drop(['branch', 'committer', 'no_src_config_edited'], axis=1, inplace=True)
    df.dropna(axis=0, subset=['last_build_result', 'time_elapse', 'time_last_failed_build', 'same_committer'], inplace=True)
    df['committer_history'] = df['committer_history'].fillna(0)
    df['committer_recent'] = df['committer_recent'].fillna(0)
    df['committer_exp'] = df['committer_exp'].fillna(0)
    df['project_history'] = df['project_history'].fillna(0)
    df['project_recent'] = df['project_recent'].fillna(0)
    df['gaussian_threat'] = df['gaussian_threat'].fillna(0)


# load feature data
def dataset(filepath):
    df = pd.read_csv(filepath)
    feature_process(df)
    col = df.columns.values.tolist()
    names = col[1:]
    y = df.iloc[:, 1].astype('int64')
    X = df.iloc[:, 2:]
    X_standard = X.apply(lambda x: (x - x.mean()) / x.std() if x.std != 0 else x)
    return X_standard, y, names


if __name__ == "__main__":
    FEATURE_DIR = "new_feature"
    project = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']

    for p in project:
        input_file = f"{FEATURE_DIR}/{p}_feature.csv"
        df = pd.read_csv(input_file)

        feature_process(df)
        OUTPUT_DIR = "new_output/data"
        output_path = f"{OUTPUT_DIR}/calibration_result.xlsx"
        writer = pd.ExcelWriter(output_path, mode='w')
        df.to_excel(writer, sheet_name='sheet1', index=False)
