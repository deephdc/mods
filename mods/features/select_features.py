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
import statsmodels.api as sm

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


# Intermittent Demand (ID)
def test_ID_levels(filename):
    print('\ntest intermittent demand levels:')

    df = utl.create_df(filename)
    print('nonzeros values: \n', df.astype(bool).sum(axis=0))

    return


# Intermittent Demand (ID)
# https://medium.com/analytics-vidhya/croston-forecast-model-for-intermittent-demand-360287a17f5f
def Croston(ts, extra_periods=1, alpha=0.4):
    d = np.array(ts)
    cols = len(d)

    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan] * extra_periods)

    # level (a), periodicity(p) and forecast (f)
    a, p, f = np.full((3, cols + extra_periods), np.nan)
    q = 1  # periods since last demand observation

    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0] / p[0]
    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = alpha * q + (1 - alpha) * p[t]
            f[t + 1] = a[t + 1] / p[t + 1]
            q = 1
        else:
            a[t + 1] = a[t]
            p[t + 1] = p[t]
            f[t + 1] = f[t]
            q += 1

    # Future Forecast
    a[cols + 1:cols + extra_periods] = a[cols]
    p[cols + 1:cols + extra_periods] = p[cols]
    f[cols + 1:cols + extra_periods] = f[cols]

    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Period": p, "Level": a, "Error": d - f})
    return df


# Intermittent Demand (ID)
# https://medium.com/analytics-vidhya/croston-forecast-model-for-intermittent-demand-360287a17f5f
def Croston_TSB(ts, extra_periods=1, alpha=0.4, beta=0.4):
    d = np.array(ts)
    cols = len(d)

    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan] * extra_periods)

    # level (a), probability(p) and forecast (f)
    a, p, f = np.full((3, cols + extra_periods), np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 / (1 + first_occurence)
    f[0] = p[0] * a[0]

    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = beta * (1) + (1 - beta) * p[t]
        else:
            a[t + 1] = a[t]
            p[t + 1] = (1 - beta) * p[t]
        f[t + 1] = p[t + 1] * a[t + 1]

    # Future Forecast
    a[cols + 1:cols + extra_periods] = a[cols]
    p[cols + 1:cols + extra_periods] = p[cols]
    f[cols + 1:cols + extra_periods] = f[cols]

    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Period": p, "Level": a, "Error": d - f})
    return df



# Autocorrelation (ACF) and partial autocorrelation (PACF)
# https://stats.stackexchange.com/questions/81754/understanding-this-acf-output
def test_autocorrelation(filename, seq_lags):
    print('\ntest_autocorrelation:')

    df = utl.create_df(filename)
    for col in list(df):
        print(col)
        ts_data = df[col]
        plt.plot(ts_data)
        plt.show()

        # autocorrelation
        print(sm.graphics.tsa.acf(ts_data, nlags=seq_lags))
        sm.graphics.tsa.plot_acf(ts_data, lags=seq_lags)
        plt.show()

        # partial autocorrelation
        print(sm.graphics.tsa.acf(ts_data, nlags=seq_lags))
        sm.graphics.tsa.plot_pacf(ts_data, lags=seq_lags)
        plt.show()
    return


# generate random walk series for testing ADF
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
            mir = mutual_info_regression(X, y, n_neighbors=7)
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
    # print(cfg.BASE_DIR)
    
    # timestamp
    timer_start = datetime.datetime.now()
    fn = cfg.app_data + cfg.data_filename_train
    # fn = cfg.app_data + cfg.data_filename_test

    # test_linearity(fn)
    # test_ID_levels(fn)
    test_autocorrelation(fn, cfg.sequence_len)
    # test_adf(fn)
    # test_mir(fn)

    # runtime report
    print('\nRuntime =', datetime.datetime.now() - timer_start)
