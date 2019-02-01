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
Created on Fri May  4 12:35:11 2018

MODS configuration file

@author: giangnguyen
@author: stefan dlugolinsky
"""

from os import path
from os.path import expanduser

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))

# Data repository
DATA_DIR = expanduser("~") + '/data/deep-dm/'  # app_data_raw
# Data dirs
dir_logs = DATA_DIR + 'logs/'
dir_parquet = DATA_DIR + 'logs_parquet/'
dir_cleaned = DATA_DIR + 'logs_cleaned/'
log_header_lines = 8

# Application dirs
app_data = BASE_DIR + '/data/'
app_data_raw = BASE_DIR + '/data/raw/'  # ln -s ...
app_data_features = BASE_DIR + '/data/features/'  # extracted features
app_models = BASE_DIR + '/models/'
app_checkpoints = BASE_DIR + '/checkpoints/'
app_visualization = BASE_DIR + '/visualization/'

MODS_RemoteStorage = 'deepnc:/Datasets/mods/'
MODS_DataDir = 'data'

# Feature data
feature_filename = 'features.tsv'
# time_range_begin = '2018-04-14'         # begin <= time_range < end
# time_range_end   = '2018-10-15'         # excluded
time_range_begin = '2018-07-14'
time_range_end = '2018-10-15'
window_duration = '1 hour'
slide_duration = '10 minutes'

# ML data
column_separator = '\t'  # for tsv
# column_separator = ','                                                    # for csv

# ML and time series datasets
split_ratio = 0.67  # train:test = 2:1
batch_size = 1  # delta (6), without_delta(1)

# Auxiliary
rate_RMSE = True

# Auxiliary: plotting
drawing = True
fig_size_x = 15  # max 2^16 pixels = 650 inch
fig_size_y = 4

# Auxiliary: DayTime format
format_string = '%Y-%m-%d %H:%M:%S'
format_string_parquet = '%Y-%m-%d %H_%M_%S'  # parquet format without ":"
timezone = 3600

# pandas defaults
pd_usecols = ['number_of_conn', 'sum_orig_kbytes']
pd_sep = '\t'
pd_skiprows = 0
pd_skipfooter = 0
pd_engine = 'python'
pd_header = 0

# training defaults
data_train = path.join(app_data_features, 'features-20180414-20181015-win-1_hour-slide-10_minutes.tsv')
multivariate = len(pd_usecols)
sequence_len = 6
model_delta = True
interpolate = True
model_types = ['LSTM', 'bidirect', 'seq2seq', 'GRU', 'CNN', 'MLP']
model_type = model_types[0]
num_epochs = 50
epochs_patience = 10
blocks = 6

# prediction defaults
data_test = path.join(app_data_features, 'features-20180414-20181015-win-1_hour-slide-10_minutes.tsv')

# common defaults
model_name = 'mods-20180414-20181015-w1h-s10m'


def set_pandas_args():
    pandas_args = {
        'pd_usecols': {
            'default': ','.join(pd_usecols),
            'help': 'A list of column names separated by comma; e.g., number_of_conn,sum_orig_kbytes',
            'required': False
        },
        # 'pd_sep': {
        #     'default': pd_sep,
        #     'help': '',
        #     'required': False
        # },
        'pd_skiprows': {
            'default': pd_skiprows,
            'help': '',
            'required': False
        },
        'pd_skipfooter': {
            'default': pd_skipfooter,
            'help': '',
            'required': False
        },
        # 'pd_engine': {
        #     'default': pd_engine,
        #     'help': '',
        #     'required': False
        # },
        'pd_header': {
            'default': pd_header,
            'help': '',
            'required': False
        }
    }
    return pandas_args


def set_train_args():
    train_args = {
        'model_name': {
            'default': model_name,
            'help': 'Name of the trained model',
            'type': str,
            'required': False
        },
        'data': {
            'default': data_train,
            'help': 'Training data to train on',
            'required': False
        },
        'multivariate': {
            'default': multivariate,
            'help': '',
            'required': False
        },
        'sequence_len': {
            'default': sequence_len,
            'help': '',
            'required': False
        },
        'model_delta': {
            'default': model_delta,
            'help': '',
            'required': False
        },
        'interpolate': {
            'default': interpolate,
            'help': '',
            'required': False
        },
        'model_type': {
            'default': model_type,
            'choices': model_types,
            'help': '',
            'required': False
        },
        'num_epochs': {
            'default': num_epochs,
            'help': 'Number of epochs to train on',
            'required': False
        },
        'epochs_patience': {
            'default': epochs_patience,
            'help': '',
            'required': False
        },
        'blocks': {
            'default': blocks,
            'help': '',
            'required': False
        }
    }
    train_args.update(set_pandas_args())
    return train_args


def set_predict_args():
    predict_args = {
        'model_name': {
            'default': model_name,
            'help': 'Name of the trained model',
            'type': str,
            'required': False
        },
        'data': {
            'default': data_train,
            'help': 'Training data to train on',
            'required': False
        }
    }
    predict_args.update(set_pandas_args())
    return predict_args
