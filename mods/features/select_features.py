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

import datetime

import numpy as np
import pandas as pd
import sympy

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

from scipy.stats import pearsonr
from scipy.stats import linregress

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def create_df(feature_filename):
    df = pd.read_csv(feature_filename,
                     sep=cfg.column_separator,
                     skiprows=0,
                     skipfooter=0,
                     engine='python',
                     usecols=lambda col: col in cfg.pd_usecols
                     )
    df.replace('None', 0, inplace=True)
    df.interpolate(inplace=True)
    print(df.shape)
    return df


def test_linearity(feature_filename):
    df = create_df(feature_filename)
    print(len(df.columns), list(df))

    reduced_form, inds = sympy.Matrix(df.values).rref()
    print('\tNon-linear columns indexes: ', inds)
    # print(reduced_form)
    # print(df.iloc[:, list(inds)])
    return


def select(feature_filename):
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.3f}'.format})
    np.random.seed(0)

    df = create_df(feature_filename)
    print(len(df.columns), list(df))

    # pandas df to numpy array
    data = df.values.astype('float32')

    # data = np.random.rand(1000, 3)
    # y = data[:, 0] + np.sin(6 * np.pi * data[:, 1]) + 0.1 * np.random.randn(1000)

    X = data
    n = X.shape[1]
    for col in range(n):
        y_mask = np.ones(n, bool)
        y_mask[col] = False

        x_mask = [i for i in range(n) if i > col]
        if x_mask:
            print('y=', col, '\t x_mask=', x_mask, '\t y_mask=', y_mask)

            y = data[:, col]
            # X = data[:, x_mask]
            # print(X.shape, X.dtype.names, "\n", X)

            # MI for continues features
            mir = mutual_info_regression(X, y)
            mir /= np.max(mir)
            print("\t MI_regression: \t", mir)
            # print("\t MI_regression: \t", mir[y_mask])

            # normalized F-test for linearity
            # https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py
            # f_test, p_val = f_regression(X, y)
            # f_test /= np.max(f_test)
            # print("f_score: \t", f_test)

    # plt.figure(figsize=(15, 5))
    # for i in range(n):
    #     plt.subplot(1, n, i+1)
    #     plt.scatter(X[:, i], y, edgecolor='blue', s=10)
    #     plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    # plt.show()
    
    return


if __name__ == "__main__":
#    print(cfg.BASE_DIR)
    
    # timestamp
    timer_start = datetime.datetime.now()

    fn = cfg.app_data_features + 'features-20180414-20181015-win-1_hour-slide-10_minutes.tsv'           #  26501 lines
    test_linearity(fn)
    select(fn)

    # runtime report
    print('\nRuntime =', datetime.datetime.now() - timer_start)

#    select(cfg.app_data_features + 'yahoo_finance_stock.csv')   
#    select(cfg.app_data_features + 'serie_mock_1h_1h.tsv')     # 3 days data (very small)
#    select(cfg.app_data_features + 'serie_mock_1h_10m.tsv')     # 3 days data (very small)
#    select(cfg.app_data_features + 'features-20180714-20181015-win-1_hour-slide-10_minutes.tsv')        #  13397 lines
#    select(cfg.app_data_features + 'features-20180414-20181015-win-10_minutes-slide-10_minutes.tsv')    #  26496 lines
#    select(cfg.app_data_features + 'features-20180414-20181015-win-1_hour-slide-10_minutes.tsv')        #  26501 lines
#    select(cfg.app_data_features + 'features-20180414-20181015-win-10_minutes-slide-1_minute.tsv')      # 264969 lines
