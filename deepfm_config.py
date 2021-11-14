#coding:utf-8
import os
import sys
import pdb
import json
import numpy as np
import pandas as pd

# 默认值
default_values = {
    "age_group": -1,
    "industry": -1,
    "job": -1,
    "marital_status": -1,
    "serv_pref_1": -1,
    "serv_pref_2": -1,
    "serv_pref_3": -1,
    "serv_pref_4": -1,
    "times_paid": 3.4661927877947294  #均值
}

# dense,sparse特征
dense_features = ['times_paid']
sparse_features = ['age_group', 'industry', 'job', 'marital_status', 'serv_pref_1', 'serv_pref_2', 'serv_pref_3', 'serv_pref_4']

def read_file(file_name):
    """读取数据, 并处理默认值"""

    data = pd.read_csv(file_name)\
        .fillna(value=default_values)\
        .drop(columns=['func_list_multilabel_1','func_list_multilabel_2','func_list_multilabel_3',
                       'func_list_multilabel_4','func_list_multilabel_5'])\
        .drop(columns=['user_id'])
    data[sparse_features] = data[sparse_features].astype('int')
    return data

def generate_dense_dict(training_data, dense_features, output_file="dense_dict.json"):
    """dense特征均值、标准差"""

    res = {}
    for feat in dense_features:
        mean_value = training_data[feat].mean()
        std_value =  training_data[feat].std()
        res[feat] = [mean_value, std_value]
    with open(output_file,"w") as fp:
        json.dump(res, fp)
    print("save {} done.".format(output_file))

def get_dense_norm(x, dense_dict, feat):
    """dense特征归一化"""

    epsilon = 1e-10
    if feat in dense_dict:
        values = dense_dict[feat]
        return (x - values[0]) / (values[1] + epsilon)
    else:
        raise LookupError("{} is not in dense_dict!".format(feat))

def generate_sparse_dict(training_data, sparse_features, output_file="sparse_dict.json"):
    """sparse特征id编码"""

    res = {}
    for feat in sparse_features:
        unique_value = [int(x) for x in sorted(training_data[feat].unique())]
        res[feat] = unique_value
    with open(output_file,"w") as fp:
        json.dump(res, fp)
    print("save {} done.".format(output_file))

def get_sparse_index(x, sparse_dict, feat):
    """sparse特征id编码"""

    if feat in sparse_dict:
        values = sparse_dict[feat]
        if x in values:
            return values.index(x)
        else:
            return len(values) #unk问题
    else:
        raise LookupError("{} is not in sparse_dict!".format(feat))

