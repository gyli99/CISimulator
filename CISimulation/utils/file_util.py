import pandas as pd
import numpy as np
import os
from datetime import datetime

# convert tr_status to 1 and 0 (1: passed, 0: broken (errored or failed))
def tr_status_convert(tr_status):
    if tr_status == 'passed':
        return 1
    elif tr_status == 'failed':
        return 0
    else:
        return np.nan

def readCIData(fileCI):
    df = pd.read_csv(fileCI)
    df['state'] = df['state'].apply(tr_status_convert)
    df.dropna(axis=0, subset=["state"], inplace=True)
    df.dropna(axis=0, subset=["started_at"], inplace=True)
    df.dropna(axis=0, subset=["finished_at"], inplace=True)

    df['started_at'] = pd.to_datetime(df['started_at'], format="%Y-%m-%dT%H:%M:%SZ")
    df['finished_at'] = pd.to_datetime(df['finished_at'], format="%Y-%m-%dT%H:%M:%SZ")

    s_date = datetime(2020, 1, 1)
    e_date = datetime(2022, 1, 1)
    df = df[(df['started_at'] >= s_date) & (df['started_at'] < e_date)]
    df.sort_values(by="started_at", inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df

def readCOData(fileCO):
    target = pd.read_csv(fileCO)
    target['date'] = pd.to_datetime(target['date'], format="%Y-%m-%dT%H:%M:%SZ")

    target['valid'] = target['date'].apply(lambda x: x.year == 2020 or x.year == 2021)
    target = target.loc[target['valid'] == True]

    target_s = target.sort_values(by="date")
    return target_s

def create_dir_if_not_exist(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)