import pandas as pd
import numpy as np
import traceback
import sys
from sklearn.metrics import confusion_matrix
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC, RandomOverSampler
from imblearn.combine import SMOTETomek,SMOTEENN
from sklearn.metrics import precision_recall_fscore_support
from mycostcla import (gen_cost_mat,
                     cla_BayesMinimumRiskClassifier,
                     cla_ThresholdingOptimization,
                     cla_CostSensitiveLogisticRegression,
                     cla_CostSensitiveDecisionTreeClassifier,
                     cla_CostSensitiveRandomForestClassifier,
                     cla_CostSensitiveBaggingClassifier,
                     cla_CostSensitivePastingClassifier,
                     cla_CostSensitiveRandomPatchesClassifier)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from predict.Config import *
import data
import logging

class no_sampling:
    def __init__(self,random_state=None):
        self.random_state = random_state

    def fit_resample(self,x_train, y_train):
        return x_train, y_train

sam_operator = {
                # Over-sampling
                "RandomOverSampler":RandomOverSampler,
                "SMOTE":SMOTE,
                "BorderlineSMOTE":BorderlineSMOTE,
                "SVMSMOTE":SVMSMOTE,
                "ADASYN":ADASYN,
                # Under-sampling
                "RandomUnderSampler":RandomUnderSampler,
                "OneSidedSelection":OneSidedSelection,
                "NeighbourhoodCleaningRule":NeighbourhoodCleaningRule,
                "NearMiss":NearMiss,
                "InstanceHardnessThreshold":InstanceHardnessThreshold,
                # Combination
                "SMOTETomek":SMOTETomek,
                "SMOTEENN":SMOTEENN,
                # no_sampling
                "no_sampling": no_sampling}


# Ensemble Classifier
def cla_BalancedRandomForestClassifier(x_train, y_train, x_test):
    print("BalancedRandomForestClassifier")
    cla = BalancedRandomForestClassifier(max_depth=5, random_state=42)
    cla.fit(x_train, y_train)
    y_pred = cla.predict(x_test)
    return y_pred

def cla_RUSBoostClassifier(x_train, y_train, x_test):
    print("RUSBoostClassifier")
    cla = RUSBoostClassifier(random_state=42)
    cla.fit(x_train, y_train)
    y_pred = cla.predict(x_test)
    return y_pred

def cla_BalancedBaggingClassifier(x_train, y_train, x_test):
    print("BalancedBaggingClassifier")
    cla = BalancedBaggingClassifier(random_state=42)
    cla.fit(x_train, y_train)
    y_pred = cla.predict(x_test)
    return y_pred

def cla_EasyEnsembleClassifier(x_train, y_train, x_test):
    print("EasyEnsembleClassifier")
    cla = EasyEnsembleClassifier(random_state=0)
    cla.fit(x_train, y_train)
    y_pred = cla.predict(x_test)
    return y_pred

# SVM
def cla_SVC(x_train, y_train, x_test):
    print("SVC")
    cla = SVC(gamma='scale')
    cla.fit(x_train,y_train)
    y_pred = cla.predict(x_test)
    return y_pred

def cla_OneClassSVM(x_train, y_train, x_test):
    x_train, y_train = x_train[y_train == 1], y_train[y_train == 1]
    print("OneClassSVM")
    cla = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    cla.fit(x_train)
    y_pred = cla.predict(x_test)
    y_pred = np.where(y_pred==-1,0,1)
    return y_pred

def cla_LocalOutlierFactor(x_train, y_train, x_test):
    cla =LocalOutlierFactor(n_neighbors=20)
    cla.fit(x_train)
    y_pred = cla.fit_predict(x_test)
    y_pred = np.where(y_pred==-1,0,1)
    return y_pred

def cla_RandomForestClassifier(x_train, y_train, x_test):
    print("RandomForestClassifier")
    cla = RandomForestClassifier(max_depth=5, random_state=42)
    cla.fit(x_train, y_train)
    y_pred = cla.predict(x_test)
    return y_pred

def cla_GaussianNB(x_train, y_train, x_test):
    print("GaussianNB")
    cla = GaussianNB()
    cla.fit(x_train, y_train)
    y_pred = cla.predict(x_test)
    return y_pred

def cla_DecisionTreeClassifier(x_train, y_train, x_test):
    print("DecisionTreeClassifier")
    cla = DecisionTreeClassifier(random_state=42)
    cla.fit(x_train, y_train)
    y_pred = cla.predict(x_test)
    return y_pred

def cla_GradientBoostingClassifier(x_train, y_train, x_test):
    print("GradientBoostingClassifier")
    cla = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=5,random_state=42)
    cla.fit(x_train, y_train)
    y_pred = cla.predict(x_test)
    return y_pred


# def classifier_operator(x_train,y_train):
cla_operator = {
                "BalancedRandomForestClassifier":cla_BalancedRandomForestClassifier,
                "EasyEnsembleClassifier":cla_EasyEnsembleClassifier,
                "RUSBoostClassifier":cla_RUSBoostClassifier,
                "BalancedBaggingClassifier":cla_BalancedBaggingClassifier,
                "SVC":cla_SVC,
                "OneClassSVM":cla_OneClassSVM,
                "LocalOutlierFactor":cla_LocalOutlierFactor,
                }

bal_cla_operator = {"RandomForestClassifier": cla_RandomForestClassifier,
                    "DecisionTreeClassifier": cla_DecisionTreeClassifier,
                    "GradientBoostingClassifier": cla_GradientBoostingClassifier,
                    "GaussianNB": cla_GaussianNB,}

cos_cla_operator = {"BayesMinimumRiskClassifier":cla_BayesMinimumRiskClassifier,
                    "ThresholdingOptimization":cla_ThresholdingOptimization,
                    "CostSensitiveLogisticRegression":cla_CostSensitiveLogisticRegression,
                    "CostSensitiveDecisionTreeClassifier":cla_CostSensitiveDecisionTreeClassifier,
                    "CostSensitiveRandomForestClassifier":cla_CostSensitiveRandomForestClassifier,
                    "CostSensitiveBaggingClassifier":cla_CostSensitiveBaggingClassifier,
                    "CostSensitivePastingClassifier":cla_CostSensitivePastingClassifier,
                    "CostSensitiveRandomPatchesClassifier":cla_CostSensitiveRandomPatchesClassifier,}


def predictor_measurement_variables(y_test, y_pred):
    def cal_helper(aa, a):
        r = np.nan
        try:
            r = aa / a
        except:
            pass
        return r
    df = pd.DataFrame()
    df['real'] = y_test
    df['predict'] = y_pred
    prev_predict = [0]
    prev_predict.extend(y_pred[0:-1])
    df['prev_predict'] = prev_predict

    df_real0 = df.loc[df['real'] == 0]
    df_real1 = df.loc[df['real'] == 1]

    fail_f = df_real0.loc[df_real0['prev_predict'] == 0].shape[0]
    fail_p = df_real0.loc[df_real0['prev_predict'] == 1].shape[0]
    fail_ff = df_real0.loc[(df_real0['predict'] == 0) & (df_real0['prev_predict'] == 0)].shape[0]
    fail_pp = df_real0.loc[(df_real0['predict'] == 1) & (df_real0['prev_predict'] == 1)].shape[0]

    pass_f = df_real1.loc[df_real1['prev_predict'] == 0].shape[0]
    pass_p = df_real1.loc[df_real1['prev_predict'] == 1].shape[0]
    pass_ff = df_real1.loc[(df_real1['predict'] == 0) & (df_real1['prev_predict'] == 0)].shape[0]
    pass_pp = df_real1.loc[(df_real1['predict'] == 1) & (df_real1['prev_predict'] == 1)].shape[0]

    fail_ff_rate = cal_helper(fail_ff, fail_f)
    fail_pp_rate = cal_helper(fail_pp, fail_p)
    pass_ff_rate = cal_helper(pass_ff, pass_f)
    pass_pp_rate = cal_helper(pass_pp, pass_p)

    return fail_ff_rate, fail_pp_rate, pass_ff_rate, pass_pp_rate


def cal_performance(y_test, y_pred):
    def cal_Fscore(precision, recall):
        if precision == 0 or recall == 0:
            F1, F2, F05 = 0, 0, 0
        else:
            F1 = 2 * precision * recall / (precision + recall)
            F2 = 5 * precision * recall / (4 * precision + recall)
            F05 = 1.25 * precision * recall / (0.25 * precision + recall)
        return F1, F2, F05
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average="binary")
    f1, f2, f05 = cal_Fscore(prec, recall)
    prec0, recall0, fscore0, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=0, average="binary")
    f1_anti, f2_anti, f05_anti = cal_Fscore(prec0, recall0)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fpr0, tpr0, thresholds0 = metrics.roc_curve(y_test, y_pred, pos_label=0)
    auc0 = metrics.auc(fpr0, tpr0)
    accuracy = (tn + tp) / (tn + tp + fn + fp)

    fail_ff_rate, fail_pp_rate, pass_ff_rate, pass_pp_rate = predictor_measurement_variables(y_test, y_pred)

    return [fail_ff_rate, fail_pp_rate, pass_ff_rate, pass_pp_rate,
            tn, fp, fn, tp,
            prec, recall, auc, f05, f1, f2,
            prec0, recall0, auc0, f05_anti, f1_anti, f2_anti,
            accuracy]


def runSplit(x, y, pro, end_idx):
    x_train, x_test = x.loc[:end_idx-100, :], x.loc[end_idx-100:end_idx, :]
    y_train, y_test = y.loc[:end_idx-100], y.loc[end_idx-100:end_idx]
    result = []
    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values
    if 0 in y_train:
        for i in sam_operator.keys():
            print (i)
            if i == 'NeighbourhoodCleaningRule' or i == 'NearMiss':
                sampler = sam_operator.get(i)()
            elif i == 'ADASYN':
                sampler = sam_operator.get(i)(sampling_strategy='minority')
            else:
                sampler = sam_operator.get(i)(random_state=42)
            try:
                sam_x_train, sam_y_train = sampler.fit_resample(x_train, y_train)
            except Exception as e:
                sys.stderr.write(traceback.format_exc())
                print('[ERROR Sampling]:%s\t[Project]:%s' % (i, pro))
                sam_x_train, sam_y_train = x_train, y_train

            for j in bal_cla_operator.keys():
                try:
                    y_pred = bal_cla_operator.get(j)(sam_x_train, sam_y_train, x_test)
                    performance = [i, j, end_idx]
                    performance.extend(cal_performance(y_test, y_pred))
                    result.append(performance)
                except Exception as e:
                    sys.stderr.write(traceback.format_exc())
                    print('[ERROR Classifer]:%s\t[Project]:%s' % (j, pro))
                    continue

        sam_x_train, sam_y_train = no_sampling(random_state=42).fit_resample(x_train, y_train)
        cost_mat_train = gen_cost_mat(len(sam_x_train))
        cost_mat_test = gen_cost_mat(len(x_test))

        for j in cla_operator.keys():
            try:
                y_pred = cla_operator.get(j)(sam_x_train, sam_y_train, x_test)
                performance = ["no_sampling", j, end_idx]
                performance.extend(cal_performance(y_test, y_pred))
                result.append(performance)
            except Exception as e:
                sys.stderr.write(traceback.format_exc())
                print('[ERROR Classifer]:%s\t[Project]:%s' % (j, pro))
                continue

        for j in cos_cla_operator.keys():
            try:
                y_pred = cos_cla_operator.get(j)(sam_x_train, sam_y_train, x_test, y_test, cost_mat_train,
                                                 cost_mat_test)
                performance = ["no_sampling", j, end_idx]
                performance.extend(cal_performance(y_test, y_pred))
                result.append(performance)
            except Exception as e:
                sys.stderr.write(traceback.format_exc())
                print('[ERROR Classifer]:%s\t[Project]:%s' % (j, pro))
                continue

        performance = ["accuracy=1", "accuracy=1", end_idx]
        performance.extend(cal_performance(y_test, y_test))
        result.append(performance)
    else:
        print('[ERROR]no 0 in y_train')
    return result


def traversal_run(pro, idxs):
    input_file = f"{FEATURE_DIR}/{pro}_feature.csv"
    init_x, init_y, names = data.dataset(input_file)
    output_file = f"{OUTPUT_DIR}/{pro}_predict_performance.xlsx"
    datas = []
    for end_idx in idxs:
        print(f"{pro}_{end_idx} begin")
        result = runSplit(init_x, init_y, pro, end_idx)
        datas.extend(result)

    columns = ['sampler', 'classifier', 'end_idx',
            'fail_ff', 'fail_pp', 'pass_ff', 'pass_pp',
            'tn', 'fp', 'fn', 'tp',
            'prec', 'recall', 'auc', 'f05', 'f1', 'f2',
            'prec0', 'recall0', 'auc0', 'f05_anti', 'f1_anti', 'f2_anti',
            'accuracy']
    df = pd.DataFrame(data=datas, columns=columns)
    df.to_excel(output_file, index=False, header=True)

def create_log(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('./' + logger_name + ".log")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    projects = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']
    lg_path = "new_output"
    lg = create_log(f"{lg_path}/log")
    input_file = f"../calibration/new_output/calibration_split_data.xlsx"
    for pro in projects:
        lg.info(f"Train {pro} Starts")
        df = pd.read_excel(input_file, sheet_name=pro)
        df['end_idx'] = df['split_index'].apply(lambda x: int(x.split('-')[1]))
        traversal_run(pro, df['end_idx'].tolist())
        lg.info(f"Train {pro} Ends")
