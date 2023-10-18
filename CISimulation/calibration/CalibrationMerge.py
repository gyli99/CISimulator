import pandas as pd
from calibration.Config import *

predictors = ['InstanceHardnessThreshold_RandomForestClassifier', 'no_sampling_BalancedBaggingClassifier',
              'no_sampling_CostSensitivePastingClassifier', 'NeighbourhoodCleaningRule_DecisionTreeClassifier',
              'no_sampling_LocalOutlierFactor']

def merge_calibration_variable(pro):
    print(pro)
    input_path1 = f"{OUTPUT_DIR}/calibration_split_data.xlsx"
    input_path2 = f"{FEATURE_PROCESS_DIR}/{pro}_predict_performance.xlsx"

    df1 = pd.read_excel(input_path1, sheet_name=pro)
    df2 = pd.read_excel(input_path2)

    df1['end_idx'] = df1['split_index'].apply(lambda x: int(x.split('-')[1]))
    df2['predictor'] = df2['sampler'] + "_" + df2['classifier']
    print(df1.shape)
    print(df2.shape)
    df2 = df2.loc[df2['predictor'].isin(predictors)]
    df2 = df2[['sampler', 'classifier', 'end_idx', 'fail_ff', 'fail_pp', 'pass_ff', 'pass_pp']]

    df3 = pd.merge(df1, df2, how='right', on='end_idx')
    return df3


if __name__ == '__main__':
    pros = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']
    output_path = f"{OUTPUT_DIR}/calibration_result.xlsx"
    writer = pd.ExcelWriter(output_path, mode='w')
    for p in pros:
        df = merge_calibration_variable(p)
        df.to_excel(writer, sheet_name=p, index=False)
    writer.close()
