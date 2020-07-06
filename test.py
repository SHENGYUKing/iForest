# -*- coding:utf-8 -*-

import generator as gen
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pyplot as plt
import time


def normalize(data_set):
    set_pro = []
    for data in data_set:
        data = (data - min(data))/(max(data)-min(data))
        set_pro.append(data)
    return set_pro


SAMPLE_NUM = 20
ANORMAL_FRACTION = 0.3
SECTION = 60

t1 = time.time()
test_x = normalize(gen.generator(SAMPLE_NUM, SECTION, ANORMAL_FRACTION))
t2 = time.time()
print("生成数据耗时: %.2f 秒" % (t2 - t1))

# load model
t3 = time.time()
model = joblib.load("./iforest_det.pkl")
test_y = model.predict(test_x)
t4 = time.time()
print("模型加载与推断耗时: %.2f 秒" % (t4 - t3))

plt.figure(1)
for i in range(0, 20):
    plt.subplot(4, 5, i + 1)
    plt.plot(test_x[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(test_y[i])
plt.show()
