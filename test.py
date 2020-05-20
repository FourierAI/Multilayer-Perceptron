#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: test.py
# @time: 2020-05-19 19:05
# @desc:
import numpy as np
import pandas as pd

data = np.genfromtxt('breast-cancer-wisconsin.data', delimiter=',')
pd_frame_ori = pd.DataFrame(data)
data = pd_frame_ori.dropna(axis=0, how='any').values
print('ID number:{}'.format(data[:10, 0]))
np.random.seed(17)
np.random.shuffle(data)
print('ID number:{}'.format(data[:10, 0]))
