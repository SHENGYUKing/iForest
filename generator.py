# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


def fun_normal(x, section):
    y = []
    for i in range(0, section):
        y.append(x)
    return np.asarray(y)


def function(x, section, offset=0):
    y = np.zeros((len(x),), dtype=float)
    for i in range(0, len(x)):
        if 0 <= x[i] <= section / 4:
            y[i] = x[i] + offset
        elif section * 3 / 4 <= x[i] <= section:
            y[i] = x[i] - section + offset
        else:
            y[i] = -1 * x[i] + section / 2 + offset
    return np.asarray(y)


def add_noise(f, snr):
    ps = np.sum(abs(f)**2)/len(f)
    pn = ps/(10**(snr/10))
    noise = np.random.randn(len(f)) * np.sqrt(pn)
    signal_add_noise = f + noise
    return signal_add_noise


def generator(sample_num, section, anormal_fraction):
    anormal_s = int(sample_num * anormal_fraction)
    normal_s = sample_num - anormal_s

    train = []
    offset = np.random.randint(1, 50, size=anormal_s)
    for i in range(0, anormal_s):
        x = np.linspace(0, section, section)
        y = function(x, section, offset[i])
        y_n = add_noise(y, 30)
        train.append(y_n.tolist())
    offset = np.random.randint(1, 50, size=normal_s)
    for i in range(0, normal_s):
        x = np.linspace(0, section, section)
        y = function(x, section, offset[i])
        # y = fun_normal(offset[i], section)
        # y_n = add_noise(y, 50)
        train.append(y.tolist())
    return np.asarray(train)
