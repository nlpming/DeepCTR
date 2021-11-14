#coding:utf-8
import os
import sys
import pdb
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import *
from sklearn.metrics import log_loss, roc_auc_score

#cpu运行
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from deepfm_seq_config import sparse_features, dense_features, varlen_sparse_features, params
from deepfm_seq_config import read_file, get_sparse_index, get_dense_norm, get_varlen_sparse_index
from deepfm_seq_train import build_deepfm_seq_model

if __name__ == "__main__":
    # 1. 输入输出路径
    data_path = "/home/czm/Public/interview_user_sex_predictor"
    test_data_path = os.path.join(data_path, 'data/train/test.csv')

    sparse_dict_path = os.path.join(data_path, 'data/deepfm_sparse_dict.json')
    dense_dict_path = os.path.join(data_path, 'data/deepfm_dense_dict.json')
    varlen_sparse_dict_path = os.path.join(data_path, 'data/deepfm_seq_varlen_sparse_dict.json')
    model_path = './record/models/deepfm_seq/sex_predict'
    log_path = './record/logs/deepfm_seq'

    test_data = read_file(test_data_path)
    target = ['label_value']

    # 2. 特征处理
    # 连续特征归一化
    dense_dict = {}
    with open(dense_dict_path) as fp:
        dense_dict = json.load(fp)
    if len(dense_dict) == 0:
        raise Exception("dense_dict length is 0")

    for feat in dense_features:
        test_data[feat] = test_data[feat].apply(get_dense_norm, args=(dense_dict, feat))

    # 离散特征id编码
    sparse_dict = {}
    with open(sparse_dict_path) as fp:
        sparse_dict = json.load(fp)
    if len(sparse_dict) == 0:
        raise Exception("sparse_dict length is 0")

    for feat in sparse_features:
        test_data[feat] = test_data[feat].apply(get_sparse_index, args=(sparse_dict, feat))

    # 序列离散特征id编码
    varlen_sparse_dict = {}
    with open(varlen_sparse_dict_path) as fp:
        varlen_sparse_dict = json.load(fp)
    if len(varlen_sparse_dict) == 0:
        raise Exception("varlen_sparse_dict length is 0")

    for feat in varlen_sparse_features:
        test_data[feat] = test_data[feat].apply(get_varlen_sparse_index, args=(varlen_sparse_dict, feat, params['maxlen']))

    # 3. 测试输入数据
    feature_names = dense_features + sparse_features
    test_model_input = {name: np.array(test_data[name].values.tolist()) for name in feature_names}

    # 4. 构建模型
    model = build_deepfm_seq_model(sparse_features, dense_features, sparse_dict, dense_dict, varlen_sparse_dict, params)

    # 5. 加载模型参数并预测
    model.load_weights(model_path)

    pred_ans = model.predict(test_model_input, batch_size=256) #预测结果为正样本概率值
    print("test LogLoss", round(log_loss(test_data[target].values, pred_ans), 6))
    print("test AUC", round(roc_auc_score(test_data[target].values, pred_ans), 6))

