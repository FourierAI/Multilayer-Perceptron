#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: neural_network.py
# @time: 2020-05-15 15:01
# @desc:

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def pre_process():
    global data, kfold_index
    one_series = pd.Series(np.ones((pd_frame_ori.shape[0])))
    pd_frame_ori.insert(10, '10', one_series)
    data = pd_frame_ori.dropna(axis=0, how='any').values
    np.random.seed(17)
    X_train = data[:, 1:-1]
    Y_train = data[:, -1]
    min_X = X_train.min(axis=0)
    max_X = X_train.max(axis=0)
    X_std = (X_train - min_X + 10 ** (-6)) / (max_X - min_X + 10 ** (-6)) * 2 - 1
    data[:, 1:-1] = X_std
    np.random.shuffle(data)
    kf = KFold(n_splits=5)
    kfold_index = kf.split(data)


def forward(num_H, x, w_1, w_2):
    return np.tanh(x.dot(w_1)).dot(w_2)


def SGD():
    global w_1, w_2
    for iter_time in range(T):
        np.random.shuffle(data)
        train_data = data[train_data_index]
        X = train_data[:, 1:-1]
        Y = train_data[:, -1]
        Tn = np.array([[1, 0] if y == 2 else [0, 1] for y in Y])

        # SGD
        for i in range(len(X)):
            tk = Tn[i]
            yk = forward(H, X[i], w_1, w_2)
            # error = np.sum(np.power((yk - tk), 2)) * 1 / 2
            # print('Current example error:{}'.format(error))

            delt_k = yk - tk
            weight_diff = w_2.dot(delt_k)
            aj = X[i].dot(w_1)
            zj = np.tanh(aj)
            delt_j = (1 - np.power(zj, 2)) * (weight_diff)

            up_wji = np.outer(X[i], delt_j)
            up_wkj = np.outer(zj, delt_k)

            w_1 = w_1 - learn_rate * up_wji
            w_2 = w_2 - learn_rate * up_wkj

    return w_1, w_2


if __name__ == "__main__":

    data = np.genfromtxt('breast-cancer-wisconsin.data', delimiter=',')
    pd_frame_ori = pd.DataFrame(data)

    # pre_process data
    pre_process()
    accuracy_list = []

    node_input = 10
    node_output = 2
    # hidden node number
    H = 20
    # iterator times
    T = 100
    w_region = np.sqrt(6 / (node_input + node_output + 1))
    # column vector

    learn_rate = 0.001

    for train_data_index, test_data_index in kfold_index:

        # initialize weight
        w_1 = np.random.uniform(-w_region, w_region, (node_input, H))
        w_2 = np.random.uniform(-w_region, w_region, (H, node_output))
        w_1, w_2 = SGD()

        test_data = data[test_data_index]

        X_test = test_data[:, 1:-1]
        Y_test = test_data[:, -1]
        Tn_test = np.array([[1, 0] if y == 2 else [0, 1] for y in Y_test])
        Y_test_predict = []
        for x_test in X_test:
            y_pre = forward(H, x_test, w_1, w_2)
            Y_test_predict.append(y_pre)
        Y_test_predict_arr = np.array(Y_test_predict)
        Y_test_pre = [[1 if y_b > 0.5 else 0 for y_b in y] for y in Y_test_predict_arr]
        Y_test_pre_arr = np.array(Y_test_pre)

        test = (Y_test_pre_arr == Tn_test)

        accuracy_count = np.sum(test[:, 0] & test[:, 1])
        accuracy = accuracy_count / len(Y_test_pre_arr)

        print(accuracy)
