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

import fnmatch
import os
from os import path
from os.path import expanduser


def list_dir(dir, pattern='*.tsv'):
    listOfFiles = os.listdir(dir)
    tsv_files = []
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            tsv_files.append(entry)
    return tsv_files


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
app_data_remote     = 'deepnc:/mods/data/'
app_data_raw        = BASE_DIR + '/data/raw/'
app_data_features   = BASE_DIR + '/data/features/'
app_data_test       = BASE_DIR + '/data/test/'
app_data_predict    = BASE_DIR + '/data/predict/'
app_data_plot       = BASE_DIR + '/data/plot/'
app_data_results    = BASE_DIR + '/data/results/'
app_models          = BASE_DIR + '/models/'
app_models_remote   = 'deepnc:/mods/models/'
app_checkpoints     = BASE_DIR + '/checkpoints/'
app_visualization   = BASE_DIR + '/visualization/'

# Feature data
feature_filename = 'features.tsv'
time_range_begin = '2018-04-14'             # begin <= time_range < end
time_range_end   = '2019-04-01'             # excluded
window_duration = '1 hour'
slide_duration  = '10 minutes'

# pandas defaults
pd_usecols = ['number_of_conn', 'sum_orig_kbytes']
# pd_usecols = ['number_of_conn']
# pd_usecols = ['sum_orig_kbytes']
pd_sep = '\t'                               # ',' for csv
pd_skiprows = 0
pd_skipfooter = 0
pd_engine = 'python'
pd_header = 0

# Datapool defaults
app_data_pool = app_data_features + 'w01h-s10m/'        # 'w10m-s01m/'
month_start_default = '201804'              # collected data starts since this month

data_filename_train = 'data_train.tsv'
data_train_begin = '201805'
data_train_end   = '201812'                 # included
data_train_excluded = []

data_filename_test  = 'data_test.tsv'
data_test_begin = '201901'
data_test_end   = '201903'                  # included
data_test_excluded = []

# training defaults
data_train_all = list_dir(app_data_features, '*.tsv')
data_train = 'features-20180414-20181015-win-1_hour-slide-10_minutes.tsv'

# Data transformation defaults
model_delta = True                          # True --> better predictions
interpolate = True
remove_peak = False                         # don't use; True --> worse predictions due to time-series nature

# Training parameters defaults
multivariate = len(pd_usecols)
sequence_len = 6                            # from 6 to 12 for w01h-s10m
steps_ahead = 1                             # number of steps steps_ahead for prediction
model_types = ['CuDNNLSTM', 'CuDNNGRU', 'Conv1D', 'MLP', 'BidirectLSTM', 'seq2seqLSTM']     # 'LSTM', 'GRU'
# model_types = ['ConvLSTM2D']
model_type = model_types[0]
num_epochs = 50
epochs_patience = 10
batch_size = 1                              # to be tested later
batch_size_test = 1                         # don't change
blocks = 6

# common defaults
model_name_all = list_dir(app_models, '*.zip')
# model_name = 'model-default'
model_name = 'mods-20180414-20181015-w1h-s10m'

# prediction defaults
data_predict = 'sample-w1h-s10m.tsv'        # can be removed later?

# test defaults
data_test = 'w1h-s10m.tsv'                  # can be removed later?

# Evaluation metrics on real values
eval_filename = 'eval.tsv'
eval_metrics = ['SMAPE', 'R2', 'COSINE']    # 'MAPE', 'RMSE'

# Plotting
plot = False
plot_filename = 'plot_data.png'
fig_size_x = 25                             # max 2^16 pixels = 650 inch
fig_size_y = 4

# Auxiliary: DayTime format
format_string = '%Y-%m-%d %H:%M:%S'
format_string_parquet = '%Y-%m-%d %H_%M_%S'     # parquet format without ":"
timezone = 3600


def set_common_args():
    common_args = {
        'bootstrap_data': {
            'default': True,
            'choices': [True, False],
            'help': 'Download data from remote datastore',
            'required': False
        }
    }
    return common_args


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
            'help': 'Name of the model to train',
            'type': str,
            'required': False
        },
        'data': {
            'default': data_train,
            'choices': data_train_all,
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
            'choices': [True, False],
            'help': '',
            'required': False
        },
        'interpolate': {
            'default': interpolate,
            'choices': [True, False],
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
        },
        'steps_ahead': {
            'default': steps_ahead,
            'help': 'Number of steps to predict ahead of current time',
            'required': False
        },
        'batch_size': {
            'default': batch_size,
            'help': '',
            'required': False
        }
    }
    train_args.update(set_pandas_args())
    train_args.update(set_common_args())
    return train_args


def set_predict_args():
    predict_args = {
        'model_name': {
            'default': model_name,
            'choices': model_name_all,
            'help': 'Name of the model used for prediction',
            'type': str,
            'required': False
        },
        'batch_size': {
            'default': batch_size_test,
            'help': '',
            'required': False
        }
    }
    predict_args.update(set_pandas_args())
    predict_args.update(set_common_args())
    return predict_args


def set_test_args():
    test_args = {
        'model_name': {
            'default': model_name,
            'help': 'Name of the model used for a test',
            'type': str,
            'required': False
        },
        'data': {
            'default': data_test,
            'help': 'Data to test on',
            'required': False
        },
        'batch_size': {
            'default': batch_size_test,
            'help': '',
            'required': False
        }
    }
    test_args.update(set_pandas_args())
    test_args.update(set_common_args())
    return test_args
