#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Created on Mon Nov  5 23:34:44 2018

@author: giangnguyen
"""

import mods.config as cfg
import mods.utils as utl

import datetime
import numpy as np
import pandas as pd
import sympy

from random import random
from pandas.plotting import autocorrelation_plot
# from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.3f}'.format})


# reduces a matrix to it's Reduced Row Echelon Form (rref)
def test_linearity(filename):
    print('\nselect_features test_linearity:')

    # data is numpy array
    data = utl.read_data(filename)
    reduced_form, inds = sympy.Matrix(data).rref()
    print('\tNon-linear columns indexes: ', inds)

    # print(reduced_form)
    # print(df.iloc[:, list(inds)])
    return


# generate random walk serie for testing ADF
def gen_random_walk(n=1000, plot=False):
    np.random.seed(1)
    y = []
    y.append(-1 if random() < 0.5 else 1)
    for i in range(1, n):
        movement = -1 if random() < 0.5 else 1
        value = y[i - 1] + movement
        y.append(value)
    if plot:
        autocorrelation_plot(y)
        plt.show()
    return y


# Augmented Dickey-Fuller test (ADF) --> is the time series stationary?
# FALSE if (p-value > 0.05) OR (if ADF > Critical value of 5% significant level)
def adf_interpretation(y, verbose=False):
    # ADF statistical test
    result = adfuller(y)

    # interpretation
    adf = result[0]
    p_value = result[1]
    for key, value in result[4].items():
        # print('\t\tCritical Values %s: %.3f' % (key, value))
        if key == '5%':
            critical_value_5percent = value
    if verbose:
        print('\tp-value: %f' % p_value)
        print('\tADF statistic: %f' % adf)
        print('\tCritical value: %f' % critical_value_5percent)

    if (p_value > 0.05) or (adf > critical_value_5percent):
        return False        # bad feature = random walk
    else:
        return True         # OK


def test_adf(filename, fn_out='select_features_adf.tsv'):
    print('select_features test_auto_corr')

    # y = gen_random_walk()
    # print('random_walk', adf_interpretation(y, True))

    df = utl.create_df(filename)
    with open(fn_out, 'w') as fout:
        for col in list(df):
            y = df[col].tolist()
            line = col + '\t' + str(adf_interpretation(y, True))
            print(line)
            fout.write(line + '\n')
    return


# Estimate mutual information (MI) for a continuous target variable
# MI can’t be negative, it is symmetric for discrete variable, MI estimation has variances
# the method is based on entropy estimation from k-nearest neighbors distances
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html
def test_mir(fn_in, fn_out='select_features_mir.tsv'):
    print('\nselect_features test_mir:')
    # np.random.seed(0)
    # data = np.random.rand(1000, 3)
    # y = data[:, 0] + np.sin(6 * np.pi * data[:, 1]) + 0.1 * np.random.randn(1000)

    df = utl.create_df(fn_in)
    data = df.values.astype('float32')
    col_names = list(df)

    X = data
    n = X.shape[1]
    with open(fn_out, 'w') as fout:
        for col in range(n):
            print(n, col, col_names[col])

            y = data[:, col]
            mir = mutual_info_regression(X, y)
            mir /= np.max(mir)

            line = str(col) + '\t' + '\t'.join([str(format(x, '.3f')) for x in mir.tolist()]) + '\t' + col_names[col]
            print(line)
            fout.write(line + '\n')

            # normalized F-test for linearity
            # https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py
            # f_test, p_val = f_regression(X, y)
            # f_test /= np.max(f_test)
            # print("f_score: \t", f_test)
    return


if __name__ == "__main__":
#    print(cfg.BASE_DIR)
    
    # timestamp
    timer_start = datetime.datetime.now()

    fn = cfg.app_data + cfg.data_filename_train
    # test_linearity(fn)
    # test_adf(fn)
    test_mir(fn)

    # runtime report
    print('\nRuntime =', datetime.datetime.now() - timer_start)