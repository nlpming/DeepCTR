#coding:utf-8
import os
import sys
import pdb
import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#加载deepctr模块
sys.path.append("../")
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')
    #debug
    #pdb.set_trace()

    #sparse特征：26个；dense特征：13个；
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # LabelEncoder用于sparse特征进行id化编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # dense特征进行0-1归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 统计sparse特征维度: 定义SparseFeat,DenseFeat对象
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    # 获取所有的特征名: (Ixxx -> dense特征, Cxxx -> sparse特征)
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 模型输入数据DataFrame对象
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 定义模型、优化器、损失函数
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_crossentropy'])

    # training过程
    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

    # predict过程
    pred_ans = model.predict(test_model_input, batch_size=256) #预测结果为正样本概率值
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))



