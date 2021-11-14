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

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepfm_config import sparse_features, dense_features, params
from deepfm_config import read_file, generate_sparse_dict, get_sparse_index, generate_dense_dict, get_dense_norm

def build_deepfm_model(sparse_features, dense_features, sparse_dict, dense_dict, params):
    """构建deepfm模型"""

    # 生成linear_feature_columns, dnn_feature_columns
    linear_feature_columns = []
    for feat in sparse_features:
        if feat != 'user_id':
            linear_feature_columns.append(SparseFeat(feat, vocabulary_size=len(sparse_dict[feat]) + 2,
                                                     embedding_dim=params["embedding_dim"]))
    for feat in dense_features:
        linear_feature_columns.append(DenseFeat(feat, 1, ))

    dnn_feature_columns = []
    for feat in sparse_features:
        if feat != 'user_id':
            dnn_feature_columns.append(SparseFeat(feat, vocabulary_size=training_data[feat].max() + 2,
                                                  embedding_dim=params["embedding_dim"]))
    for feat in dense_features:
        dnn_feature_columns.append(DenseFeat(feat, 1, ))

    model = DeepFM(linear_feature_columns, dnn_feature_columns,
                   task='binary',
                   dnn_hidden_units=params['dnn_hidden_units'],
                   l2_reg_linear=params['l2_reg_linear'],
                   l2_reg_embedding=params['l2_reg_embedding'],
                   l2_reg_dnn=params['l2_reg_dnn'],
                   dnn_dropout=params['dnn_dropout'],
                   dnn_activation=params['dnn_activation'],
                   dnn_use_bn=params['dnn_use_bn'])
    return model

if __name__ == "__main__":
    # 1. 输入输出路径
    data_path = "/home/czm/Public/interview_user_sex_predictor"
    training_data_path = os.path.join(data_path, 'data/train/train.csv')
    valid_data_path = os.path.join(data_path, 'data/train/valid.csv')
    sparse_dict_path = os.path.join(data_path, 'data/deepfm_sparse_dict.json')
    dense_dict_path = os.path.join(data_path, 'data/deepfm_dense_dict.json')
    model_path = './record/models/deepfm'
    log_path = './record/logs/deepfm'

    training_data = read_file(training_data_path)
    valid_data = read_file(valid_data_path)
    target = ['label_value']

    # 2. 特征处理
    # 连续特征归一化
    dense_dict = {}
    generate_dense_dict(training_data, dense_features, output_file=dense_dict_path)
    with open(dense_dict_path) as fp:
        dense_dict = json.load(fp)
    if len(dense_dict) == 0:
        raise Exception("dense_dict length is 0")

    for feat in dense_features:
        training_data[feat] = training_data[feat].apply(get_dense_norm, args=(dense_dict, feat))
        valid_data[feat] = valid_data[feat].apply(get_dense_norm, args=(dense_dict, feat))

    # 离散特征id编码
    sparse_dict = {}
    generate_sparse_dict(training_data, sparse_features, output_file=sparse_dict_path)
    with open(sparse_dict_path) as fp:
        sparse_dict = json.load(fp)
    if len(sparse_dict) == 0:
        raise Exception("sparse_dict length is 0")

    for feat in sparse_features:
        training_data[feat] = training_data[feat].apply(get_sparse_index, args=(sparse_dict, feat))
        valid_data[feat] = valid_data[feat].apply(get_sparse_index, args=(sparse_dict, feat))

    # 3. 构建模型输入数据
    feature_names = sparse_features + dense_features
    train_model_input = {name: np.array(training_data[name].values.tolist()) for name in feature_names}
    valid_model_input = {name: np.array(valid_data[name].values.tolist()) for name in feature_names}

    # 4. 构建模型
    model = build_deepfm_model(sparse_features, dense_features, sparse_dict, dense_dict, params)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
               loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.AUC()])

    # call back function
    checkpoint = ModelCheckpoint(
	    model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(
	    monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
    earlystopping = EarlyStopping(
	    monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='auto')
    tensorboard = TensorBoard(
            log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True, update_freq=100)
    callbacks = [checkpoint, reduce_lr, earlystopping, tensorboard]

    # 5. training过程
    model.fit(train_model_input, training_data[target].values,
        validation_data=(valid_model_input, valid_data[target].values),
        batch_size=params['batch_size'], epochs=params['epochs'], verbose=2, shuffle=True, callbacks=callbacks)

    # 6. predict过程
    model.load_weights(model_path)

    pred_ans = model.predict(valid_model_input, batch_size=params['batch_size']) #预测结果为正样本概率值
    print("valid LogLoss", round(log_loss(valid_data[target].values, pred_ans), 6))
    print("valid AUC", round(roc_auc_score(valid_data[target].values, pred_ans), 6))

