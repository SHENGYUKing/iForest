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


SAMPLE_NUM = 10000
ANORMAL_FRACTION = 0.15
SECTION = 60

rng = np.random.RandomState(42)

t1 = time.time()
dataset = normalize(gen.generator(SAMPLE_NUM, SECTION, ANORMAL_FRACTION))
t2 = time.time()
print("生成数据耗时: %.2f 秒" % (t2 - t1))

t3 = time.time()
train, test = train_test_split(dataset, test_size=0.3, random_state=rng)
model = IsolationForest(
    max_samples=len(train),
    max_features=SECTION,
    contamination=ANORMAL_FRACTION
)
model.fit(train)
y_pred_train = model.predict(train)
y_pred_test = model.predict(test)
t4 = time.time()
print("训练模型耗时: %.2f 秒" % (t4 - t3))
detector = joblib.dump(model, "./iforest_det.pkl")

plt.figure(1)
for i in range(0, 50):
    plt.subplot(5, 10, i + 1)
    plt.plot(test[i+2000])
    plt.xticks([])
    plt.yticks([])
    plt.title(y_pred_test[i+2000])
plt.show()
