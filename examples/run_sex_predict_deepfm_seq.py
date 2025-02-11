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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#cpu运行
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#加载deepctr模块
sys.path.append("../")
from deepctr.models import DeepFMSeq
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names

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
sparse_features = ['age_group', 'industry', 'job', 'marital_status', 'serv_pref_1',
       'serv_pref_2', 'serv_pref_3', 'serv_pref_4']
varlen_sparse_features = ['func_list_multilabel_1', 'func_list_multilabel_2', 'func_list_multilabel_3',
                          'func_list_multilabel_4', 'func_list_multilabel_5']

def read_file(file_name):
    """读取数据, 并处理默认值"""

    data = pd.read_csv(file_name)\
        .fillna(value=default_values)\
        .drop(columns=['user_id'])
    data[sparse_features] = data[sparse_features].astype('int')

    for feat in varlen_sparse_features:
        data[feat] = data[feat].apply(varlen_sparse_feature_process) #序列特征处理
    return data

def generate_sparse_index_dict(training_data, sparse_features, output_file="sparse_index_dict.json"):
    """sparse特征id编码"""

    res = {}
    for feat in sparse_features:
        unique_value = [int(x) for x in sorted(training_data[feat].unique())]
        res[feat] = unique_value
    with open(output_file,"w") as fp:
        json.dump(res, fp)
    print("save {} done.".format(output_file))

def get_sparse_index(x, sparse_index_dict, feat):
    """sparse特征id编码"""

    if feat in sparse_index_dict:
        values = sparse_index_dict[feat]
        if x in values:
            return values.index(x)
        else:
            return len(values) #unk问题
    else:
        raise LookupError("{} is not in sparse_index_dict!".format(feat))

def varlen_sparse_feature_process(x):
    """序列特征的处理"""

    res = []
    try:
        items = eval(x)
        for item in items:
            if item is np.nan:
                continue
            if item not in res:
                res.append(item)
        if len(res) != 0:
            return res
        else:
            return [-1]
    except:
        return [-1]

def generate_varlen_sparse_index_dict(training_data, varlen_sparse_features, output_file="varlen_sparse_index_dict.json"):
    res = {}
    for feat in varlen_sparse_features:
        data = list(training_data[feat].explode().unique())
        unique_value = [int(x) for x in sorted(data)]
        res[feat] = unique_value
    with open(output_file,"w") as fp:
        json.dump(res, fp)
    print("save {} done.".format(output_file))

def get_varlen_sparse_index(x, varlen_sparse_index_dict, feat, maxlen):
    """序列特征对应idx"""
    if feat in varlen_sparse_index_dict:
        values = varlen_sparse_index_dict[feat]
        res = []
        for idx in x: #list
            if idx in values:
                res.append(values.index(idx)+1)
            else:
                res.append(len(values)) #unk问题

        #maxlen长度
        if len(res) >= maxlen:
            res = res[0:maxlen]
        else:
            res += [0]*(maxlen - len(res)) #未达到maxlen，补充0;
        return np.array(res)
    else:
        raise LookupError("{} is not in sparse_index_dict!".format(feat))

if __name__ == "__main__":
    # DeepFM网络超参数
    params = {
        "embedding_dim": 10,
        "dnn_hidden_units": (20, 20, 20),
        "dnn_activation": "relu",
        "dnn_use_bn": False,
        "dnn_dropout": 0.0,
        "l2_reg_linear": 0.00001,
        "l2_reg_embedding": 0.00001,
        "l2_reg_dnn": 0,
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 20,
        "maxlen": 15
    }

    # 输入输出路径
    training_data_path = '/home/czm/Public/interview_user_sex_predictor/data/train/train.csv'
    valid_data_path = '/home/czm/Public/interview_user_sex_predictor/data/train/valid.csv'
    test_data_path = '/home/czm/Public/interview_user_sex_predictor/data/train/test.csv'
    sparse_index_dict_path = '/home/czm/Public/interview_user_sex_predictor/data/sparse_index_dict.json'
    varlen_sparse_index_dict_path = '/home/czm/Public/interview_user_sex_predictor/data/varlen_sparse_index_dict.json'
    model_path = './models/deepfm_seq'
    log_path = './logs/deepfm_seq'

    training_data = read_file(training_data_path)
    valid_data = read_file(valid_data_path)
    test_data = read_file(test_data_path)
    target = ['label_value']

    # 连续特征归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(training_data[dense_features])

    training_data[dense_features] = mms.transform(training_data[dense_features])
    valid_data[dense_features] = mms.transform(valid_data[dense_features])
    test_data[dense_features] = mms.transform(test_data[dense_features])

    # 离散特征id编码
    sparse_index_dict = {}
    generate_sparse_index_dict(training_data, sparse_features, output_file=sparse_index_dict_path)
    with open(sparse_index_dict_path) as fp:
        sparse_index_dict = json.load(fp)
    if len(sparse_index_dict) == 0:
        raise Exception("sparse_index_dict length is 0")

    for feat in sparse_features:
        training_data[feat] = training_data[feat].apply(get_sparse_index, args=(sparse_index_dict, feat))
        valid_data[feat] = valid_data[feat].apply(get_sparse_index, args=(sparse_index_dict, feat))
        test_data[feat] = test_data[feat].apply(get_sparse_index, args=(sparse_index_dict, feat))

    # 序列离散特征id编码
    varlen_sparse_index_dict = {}
    generate_varlen_sparse_index_dict(training_data, varlen_sparse_features, output_file=varlen_sparse_index_dict_path)
    with open(varlen_sparse_index_dict_path) as fp:
        varlen_sparse_index_dict = json.load(fp)
    if len(varlen_sparse_index_dict) == 0:
        raise Exception("varlen_sparse_index_dict length is 0")

    for feat in varlen_sparse_features:
        training_data[feat] = training_data[feat].apply(get_varlen_sparse_index, args=(varlen_sparse_index_dict, feat, params['maxlen']))
        valid_data[feat] = valid_data[feat].apply(get_varlen_sparse_index, args=(varlen_sparse_index_dict, feat, params['maxlen']))
        test_data[feat] = test_data[feat].apply(get_varlen_sparse_index, args=(varlen_sparse_index_dict, feat, params['maxlen']))

    # 统计sparse特征维度: 定义SparseFeat,DenseFeat对象
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=training_data[feat].max() + 2, embedding_dim=params["embedding_dim"])
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]
    varlen_sparse_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=len(varlen_sparse_index_dict[feat])+2,
                                                                 embedding_dim=params["embedding_dim"], embedding_name=feat),
                                                                 maxlen=params["maxlen"]) for feat in varlen_sparse_features]

    dnn_feature_columns = fixlen_feature_columns + varlen_sparse_feature_columns #dnn加入序列特征
    linear_feature_columns = fixlen_feature_columns

    # 获取所有的特征名
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = {name: np.array(training_data[name].values.tolist()) for name in feature_names}
    valid_model_input = {name: np.array(valid_data[name].values.tolist()) for name in feature_names}
    test_model_input = {name: np.array(test_data[name].values.tolist()) for name in feature_names}

    # 定义模型、优化器、损失函数
    model = DeepFMSeq(linear_feature_columns, dnn_feature_columns,
                   task='binary',
                   dnn_hidden_units=params['dnn_hidden_units'],
                   l2_reg_linear=params['l2_reg_linear'],
                   l2_reg_embedding=params['l2_reg_embedding'],
                   l2_reg_dnn=params['l2_reg_dnn'],
                   dnn_dropout=params['dnn_dropout'],
                   dnn_activation=params['dnn_activation'],
                   dnn_use_bn=params['dnn_use_bn'])

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
            log_dir=os.path.join(log_path,"deepfm"), histogram_freq=0, write_graph=True, write_images=True, update_freq=100)
    callbacks = [checkpoint, reduce_lr, earlystopping, tensorboard]

    # training过程
    history = model.fit(train_model_input, training_data[target].values,
                        validation_data=(valid_model_input, valid_data[target].values),
                        batch_size=params['batch_size'], epochs=params['epochs'], verbose=2, shuffle=True, callbacks=callbacks)

    # predict过程
    model.load_weights(model_path)
    pred_ans = model.predict(test_model_input, batch_size=params['batch_size']) #预测结果为正样本概率值
    print("test LogLoss", round(log_loss(test_data[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test_data[target].values, pred_ans), 4))



