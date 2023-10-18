import pandas as pd
import numpy as np
import traceback
import sys
from collections import Counter
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from costcla.metrics import savings_score
from sklearn.metrics import classification_report
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
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from predict.Config import *
import data

def cal_avg_performance(inputfile, outputfile, pro):
    model_dic = {}

    df = pd.read_csv(inputfile)
    for index, row in df.iterrows():
        sampler = row['Sampler']
        classifer = row['Classifer']
        precision = row['Precision']
        recall = row['Recall']
        f05 = row['F05']
        f1 = row['F1']
        f2 = row['F2']

        precision0 = row['Precision0']
        recall0 = row['Recall0']
        f05_anti = row['F05_anti']
        f1_anti = row['F1_anti']
        f2_anti = row['F2_anti']

        auc = row['AUC']
        auc0 = row['AUC0']
        accuracy = row['Accuracy']

        if (sampler, classifer) in model_dic.keys():
            model_dic[(sampler, classifer)].append([precision, recall, f05, f1, f2,
                                                    precision0, recall0, f05_anti, f1_anti, f2_anti,
                                                    auc, auc0, accuracy])
        else:
            model_dic[(sampler, classifer)] = []
            model_dic[(sampler, classifer)].append([precision, recall, f05, f1, f2,
                                                    precision0, recall0, f05_anti, f1_anti, f2_anti,
                                                    auc, auc0, accuracy])

    result_df = pd.DataFrame(columns=['Sampler', 'Classifer', 'AVG_Precision', 'AVG_Recall', 'AVG_F05', 'AVG_F1', 'AVG_F2',
                                      'AVG_Precision0', 'AVG_Recall0', 'AVG_F05_anti', 'AVG_F1_anti', 'AVG_F2_anti',
                                      'AVG_AUC', 'AVG_AUC0', 'AVG_Accuracy'])
    for key, value in model_dic.items():
        tmp_df = pd.DataFrame(data=value,
                              columns=['AVG_Precision', 'AVG_Recall', 'AVG_F05', 'AVG_F1', 'AVG_F2',
                                      'AVG_Precision0', 'AVG_Recall0', 'AVG_F05_anti', 'AVG_F1_anti', 'AVG_F2_anti',
                                      'AVG_AUC', 'AVG_AUC0', 'AVG_Accuracy'])
        tmp_result = tmp_df.mean()
        tmp_std = tmp_df.std(ddof=0)

        for col in tmp_std.index.to_list():
            std_col = col.replace('AVG', 'STD')
            tmp_result[std_col] = tmp_std[col]

        tmp_result['Sampler'] = key[0]
        tmp_result['Classifer'] = key[1]

        result_df = result_df.append(tmp_result, ignore_index=True)

    result_df = result_df[['Sampler', 'Classifer',
                           'AVG_Precision', 'AVG_Recall', 'AVG_F05', 'AVG_F1', 'AVG_F2',
                            'AVG_Precision0', 'AVG_Recall0', 'AVG_F05_anti', 'AVG_F1_anti', 'AVG_F2_anti',
                            'AVG_AUC', 'AVG_AUC0', 'AVG_Accuracy',
                           'STD_Precision', 'STD_Recall', 'STD_F05', 'STD_F1', 'STD_F2',
                           'STD_Precision0', 'STD_Recall0', 'STD_F05_anti', 'STD_F1_anti', 'STD_F2_anti',
                           'STD_AUC', 'STD_AUC0', 'STD_Accuracy']]
    writer = pd.ExcelWriter(outputfile, mode='w')
    result_df.to_excel(writer, sheet_name=pro, index=False)

class no_sampling:
    def __init__(self,random_state=None):
        self.random_state = random_state

    def fit_resample(self,x_train, y_train):
        return x_train, y_train

sam_operator_used = {
    "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule,
    "InstanceHardnessThreshold": InstanceHardnessThreshold,
    "no_sampling": no_sampling
}

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

#SVM
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

samplers = [NeighbourhoodCleaningRule, InstanceHardnessThreshold, no_sampling, no_sampling, no_sampling]
operators = [cla_DecisionTreeClassifier, cla_RandomForestClassifier, cla_BalancedBaggingClassifier,
             cla_LocalOutlierFactor, cla_CostSensitivePastingClassifier]
NAMES = ['NCR+DT', 'IHT+RF', 'BB', 'LOF', 'CSP']

def append_results(numsplit,samname,claname,y_test,y_pred,
                   NumSplit,SamName,ClaName,Precision,Recall,
                   Fscore05, Fscore1, Fscore2, Precision0, Recall0,
                   Fscore05_anti, Fscore1_anti, Fscore2_anti,TN,FP,FN,TP,AUC,AUC0, Accuracy):
    def cal_Fscore(precision, recall):
        if precision == 0 or recall == 0:
            F1, F2, F05 = 0, 0, 0
        else:
            F1 = 2 * precision * recall / (precision + recall)
            F2 = 5 * precision * recall / (4 * precision + recall)
            F05 = 1.25 * precision * recall / (0.25 * precision + recall)
        return F1, F2, F05

    # classifier.fit(sam_x_train, sam_y_train,x_test)
    # y_pred = classifier.predict(x_test)
    # target_names = ['failed', 'passed']  # doctest : +NORMALIZE_WHITESPACE
    # report = classification_report_imbalanced(y_test, y_pred, target_names=target_names)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average="binary")
    f1, f2, f05 = cal_Fscore(prec, recall)
    prec0, recall0, fscore0, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=0, average="binary")
    f1_anti, f2_anti, f05_anti = cal_Fscore(prec0, recall0)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fpr0, tpr0, thresholds0 = metrics.roc_curve(y_test, y_pred, pos_label=0)
    auc0 = metrics.auc(fpr0, tpr0)
    accuracy = (tn + tp) / (tn + tp + fn + fp)

    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)
    Precision.append(prec)
    Recall.append(recall)
    Fscore05.append(f05)
    Fscore1.append(f1)
    Fscore2.append(f2)
    Precision0.append(prec0)
    Recall0.append(recall0)
    Fscore05_anti.append(f05_anti)
    Fscore1_anti.append(f1_anti)
    Fscore2_anti.append(f2_anti)
    AUC.append(auc)
    AUC0.append(auc0)
    Accuracy.append(accuracy)
    NumSplit.append(numsplit)
    SamName.append(samname)  # record sampler name
    ClaName.append(claname)  # record classifier name

def runtimeSeriesSplit(x, y, pro):
    output_path = f'../predict/s_output/{pro}'

    predict = []
    real = []
    r = []

    n_splits = round(len(x)/1000) - 1
    kf = TimeSeriesSplit(n_splits=n_splits)
    round_count = 0
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if 0 in y_train:
            for i in range(samplers.__len__()):
                if i == 0:
                    sampler = samplers[i]()
                else:
                    sampler = samplers[i](random_state=42)
                try:
                    sam_x_train, sam_y_train = sampler.fit_resample(x_train, y_train)
                except Exception as e:
                    sys.stderr.write(traceback.format_exc())
                    print('[ERROR Sampling]:%s\t[Project]:%s' % (i, pro))
                    sam_x_train, sam_y_train = x_train, y_train

                if i == 0 or i == 1:
                    y_pred = operators[i](sam_x_train, sam_y_train, x_test)
                else:
                    sam_x_train, sam_y_train = no_sampling(random_state=42).fit_resample(x_train, y_train)
                    cost_mat_train = gen_cost_mat(len(sam_x_train))
                    cost_mat_test = gen_cost_mat(len(x_test))
                    if i == 2 or i == 3:
                        y_pred = operators[i](sam_x_train, sam_y_train, x_test)
                    else:
                        y_pred = operators[i](sam_x_train, sam_y_train, x_test, y_test, cost_mat_train, cost_mat_test)

                for j in range(y_pred.__len__()):
                    predict.append(y_pred[j])
                    real.append(y_test[j])
                    r.append(round_count)

                df = pd.DataFrame(columns=[])
                writer = pd.ExcelWriter(output_path, mode='a')
                df.insert(0, 'predict', predict)
                df.insert(1, 'real', real)
                df.insert(3, 'round', r)
                df.to_excel(writer, sheet_name=NAMES[i], index=False, header=True)
                writer.close()
                round_count += 1
        else:
            print('[ERROR]no 0 in y_train')
            continue

def traversal_run(projects):
    for pro in projects:
        print('%s begin' % pro)
        input_file = f"{FEATURE_DIR}/{pro}_feature.csv"
        init_x, init_y, names = data.dataset(input_file)
        runtimeSeriesSplit(init_x, init_y, pro)

def traversal_avg_performance_run(projects):
    for pro in projects:
        print('%s begin' % pro)
        input_file = f"{OUTPUT_DIR}/time_validation_results.xlsx"
        output_file = f"{OUTPUT_DIR}/time_validation_results_avg.xlsx"
        cal_avg_performance(input_file, output_file, pro)


if __name__ == "__main__":
    projects = ['python@cpython', 'pypa@warehouse', 'apache@hive', 'pypa@pip', 'akka@akka', 'opf@openproject']
    traversal_run(projects)
