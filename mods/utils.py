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
Created on Mon Apr 23 12:48:52 2018

@author: giangnguyen
@author: stefan dlugolinsky
"""

import calendar
import datetime
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import time
# from datetime import datetime
from datetime import timedelta
from math import sqrt
from os.path import basename
from random import randint
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import keras

# from mods import config as mc
import mods.config as cfg

matplotlib.style.use('ggplot')


# %matplotlib inline

# import bat
# import pyarrow
# import pyspark
# def print_libs_versions():
#    print('BAT: {:s}'.format(bat.__version__))
#    print('Numpy: {:s}'.format(np.__version__))
#    print('Pandas: {:s}'.format(pd.__version__))
#    print('PyArrow: {:s}'.format(pyarrow.__version__))
#    print('PySpark: {:s}'.format(pyspark.__version__))
#    print('Scikit Learn: ', sklearn.__version__)
#    return


########## Data functions ##########

# @giang: get Y or X from TimeseriesGenerator, default is Y
def get_var_from_tsg(tsg, multivariate, get_x=False, verbose=False):
    var = None

    index = 1
    if get_x:
        index = 0

    for i in range(len(tsg)):
        batch = tsg[i][index]
        if var is None:
            var = batch
        else:
            var = np.append(var, batch)

    if verbose:
        print(var.shape, '\n', var)

    var = var.reshape((-1, multivariate))

    if verbose:
        print(var.shape)

    return var


# @giang: First order differential d(y)/d(t)=f(y,t)=y' for numpy array
def delta_timeseries(arr):
    return arr[1:] - arr[:-1]


# @giang: RMSE for numpy array
def rate_rmse(a, b):
    score = []
    for i in range(a.shape[1]):
        score.append(sqrt(mean_squared_error(a[:, i], b[:, i])))
    return score


# @giang: cosine similarity for two numpy arrays
def rate_cosine(a, b):
    score = []
    for i in range(a.shape[1]):
        score.append(cosine_similarity(a[:, i].reshape(1, -1),
                                       b[:, i].reshape(1, -1)))
    return score


########## File manipulation functions ##########

# @giang create unique filename
def create_filename(
        timer_start,
        dir_output=cfg.app_data_features,
        feature_filename=cfg.feature_filename,
        time_range_begin=cfg.time_range_begin,
        time_range_end=cfg.time_range_end,
        window_duration=cfg.window_duration,
        slide_duration=cfg.slide_duration
):
    filename = os.path.join(
        dir_output,
        cfg.feature_filename.split('.')[0] + \
        '-%s-%s-win-%s-slide-%s-ts-%s.tsv' % (
            re.sub(r'[-\s:]+', r'', str(time_range_begin)),  # without %H:%M:%S
            re.sub(r'[-\s:]+', r'', str(time_range_end)),
            re.sub(r'\s+', r'_', window_duration),
            re.sub(r'\s+', r'_', slide_duration),
            str(int(timer_start.timestamp())))
    )
    print('Output filename for extracted features:', filename)
    return filename


# @giang
def get_fullpath_model_name(dataset_name,
                            sequence_len=cfg.sequence_len):
    model_name = (cfg.app_models +
                  os.path.splitext(basename(dataset_name))[0] +
                  '-seq-' + str(sequence_len) + '.h5')
    return model_name


########## Drawing functions ##########

# @giang data = numpy array
def plot_series(data, ylabel):
    plt.plot(data)
    plt.ylabel(ylabel)
    plt.show()


# df = pandas dataframe
def drawing_df_scaled():
    np.random.seed(1)
    scaler = MinMaxScaler(feature_range=(0, 1))

    df = pd.DataFrame({
        'x1': np.random.normal(0, 2, 10000),
        'x2': np.random.normal(5, 3, 10000),
        'x3': np.random.normal(-5, 5, 10000)
    })
    #    df = pd.DataFrame({
    #        'x1': np.random.chisquare(8, 1000),
    #        'x2': np.random.beta(8, 2, 1000) * 40,
    #        'x3': np.random.normal(50, 3, 1000)
    #    })

    df_scaled = scaler.fit_transform(df)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

    ax1.set_title('Before scaling')
    for col in list(df):
        sns.kdeplot(df[col].values, ax=ax1)

    ax2.set_title('After scaling')
    for col in list(df_scaled):
        sns.kdeplot(df_scaled[col].values, ax=ax2)

    plt.show()
    return


########## DayTime functions ##########

# @stevo
REGEX_TIME_INTERVAL = re.compile(
    r'((?P<years>\d)\s+years?\s+)?((?P<months>\d)\s+months?\s+)?((?P<days>\d)\s+days?\s+)?(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})(?P<nanoseconds>\.\d+)')


# @stevo
def parseInterval(s):
    global REGEX_TIME_INTERVAL
    time_array = [['nanoseconds', 1],
                  ['seconds', 1],
                  ['minutes', 60],
                  ['hours', 3600],
                  ['days', 86400],
                  ['months', 1036800],
                  ['years', 378432000]]

    m = REGEX_TIME_INTERVAL.search(s.strip())
    seconds = float(0.0)
    for t in time_array:
        seconds += float(m.group(t[0])) * t[1] if m.group(t[0]) else 0
    return seconds


########## Auxiliary functions - can be removed later ##########

def get_one_row(i, dataset):
    row = dataset[i, :]
    return row[1:].reshape(1, -1)  # i-th row without label as [row]


def get_random_row(dataset):
    i = randint(0, dataset.shape[0])
    return get_one_row(i, dataset)


def load_dataset(dataset_name):
    dataset = np.loadtxt(cfg.app_data + dataset_name, delimiter=',')
    print(dataset_name, dataset.shape)
    return dataset


# [range_0, ..., range_n]
def get_range(start, end, window=cfg.window_duration):
    ss = str(window) + 'S'
    return pd.date_range(start, end, freq=ss).floor(ss)


# convert "2018-10-05 21:47:57" to (int)timestamp 
def string_to_timestamp(timer_string, fs=cfg.format_string):
    ts = time.mktime(datetime.strptime(timer_string, fs).timetuple())
    return ts


# convert timestamp to string in format
def unixtime_to_string(ts, fs=cfg.format_string):
    return int(datetime.utcfromtimestamp(ts).strftime(fs))


# '2018-05' --> '2018-05-01'
def get_month_start(ym='2018-05'):
    return ym + '-01'


# '2018-08' --> '2018-08-31'
def get_month_end(ym='2018-09'):
    ss = ym.split('-')
    return ym + '-' + str(calendar.monthrange(int(ss[0]), int(ss[1]))[1])


# Slovakia GTM+2 = 7200 seconds
def window_start(dir_day, log_file,
                 timezone=cfg.timezone,
                 fs=cfg.format_string_parquet):
    dt = dir_day + ' ' + log_file.split('.')[1].split('-')[0]
    dto = datetime.strptime(dt, fs) + timedelta(seconds=timezone)
    start = dto.strftime(fs)
    # print(start)    
    return start


# slide window 3600s
def window_next(start, window_duration=cfg.window_duration,
                fs=cfg.format_string_parquet):
    dto = datetime.strptime(start, fs) + timedelta(seconds=window_duration)
    end = dto.strftime(fs)
    # end = time.mktime(dto.timetuple())           # returns seconds
    # print(end)
    return end


# 2018-05-01 00:00:00
def print_slides(dir_day, log_file, window_duration=cfg.window_duration,
                 fs=cfg.format_string):
    start = window_start(dir_day, log_file, window_duration, fs)
    for i in range(0, int(3600 / window_duration) + 1):
        print('\tslide', i,
              window_next(start, window_duration * i, fs=cfg.format_string))
    return


# returns metadata filename based on model filename
def get_metadata_filename(model_filename):
    return re.sub(r'\.[^.]+$', r'.json', model_filename)


# loads model and model's metadata
def load_model(filename, metadata_filename):
    try:
        model = keras.models.__load_model(filename)
        metadata = load_model_metadata(metadata_filename)
        return model, metadata
    except Exception as e:
        print(e)
        return None


# loads and returns model metadata
def load_model_metadata(metadata_filename):
    try:
        with open(metadata_filename, 'rb') as f:
            return json.load(f)
    except Exception as e:
        print(e)
    return None
