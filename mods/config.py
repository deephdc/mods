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

import datetime
import fnmatch
import logging
import os
import pathlib
from mods.mods_types import TimeRange


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def list_dir(dir, pattern='*.tsv'):
    tsv_files = []
    try:
        listOfFiles = os.listdir(dir)
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry, pattern):
                tsv_files.append(entry)
    except OSError as e:
        print(e)
    return tsv_files


# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

#default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        logging.info(msg)

REMOTE_BASE_DIR = IN_OUT_BASE_DIR
if 'APP_REMOTE_BASE_DIR' in os.environ:
    env_remote_base_dir = os.environ['APP_REMOTE_BASE_DIR']
    if os.path.isdir(env_remote_base_dir):
        REMOTE_BASE_DIR = env_remote_base_dir
    else:
        msg = "[WARNING] \"APP_REMOTE_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_remote_base_dir) + \
        "Using \"IN_OUT_BASE_DIR={}\" instead.".format(IN_OUT_BASE_DIR)
        logging.info(msg)

# Application dirs
app_data_remote     = os.path.join(REMOTE_BASE_DIR, 'data')
app_models_remote   = os.path.join(REMOTE_BASE_DIR, 'models')
app_data            = os.path.join(IN_OUT_BASE_DIR, 'data')
app_data_features   = os.path.join(app_data, 'tsv')
app_data_test       = os.path.join(app_data, 'test')
app_models          = os.path.join(IN_OUT_BASE_DIR, 'models')
app_checkpoints     = os.path.join(IN_OUT_BASE_DIR, 'checkpoints')
app_cache           = os.path.join(IN_OUT_BASE_DIR, 'cache')
app_data_pool_cache = os.path.join(app_cache, 'datapool')
app_tensordboard    = os.path.join(IN_OUT_BASE_DIR, 'tensorboard')

#pathlib.Path(app_data).mkdir(parents=True, exist_ok=True)
#pathlib.Path(app_models).mkdir(parents=True, exist_ok=True)
pathlib.Path(app_checkpoints).mkdir(parents=True, exist_ok=True)
pathlib.Path(app_cache).mkdir(parents=True, exist_ok=True)

# Generic settings
time_range_inclusive_beg = True  # True: <beg; False: (beg
time_range_inclusive_end = True  # True: end>; False: end)

# Datapool defaults
app_data_pool = app_data_features + 'w01h-s10m/'        # 'w10m-s01m/'
data_pool_caching = True

# !!! column names must be distinct (use tilde (~) to rename column; e.g., orig_col_name~new_col_name !!!
# TODO: NaN problem: 'sip|internal_count_uid~sip_in;' +\
data_select_query = \
    'conn|in_count_uid~conn_in|out_count_uid~conn_out;' + \
    'dns|in_distinct_query~dns_in_distinct;' + \
    'ssh|in~ssh_in' + \
    '#window_start,window_end'

# Datapools: window-slide
ws_choices = ['w01h-s10m', 'w10m-s01m', 'w10m-s10m']
ws_choice = ws_choices[0]

# Training parameters defaults
train_data_select_query = data_select_query
model_delta = True                          # True --> better predictions, first order differential
sequence_len = 12                           # p in <6, 24> for w01h-s10m
steps_ahead = 1                             # k in <1, 12> for w01h-s10m; k < p
model_types = [
    'MLP',
    'Conv1D',
    'autoencoderMLP',
    'LSTM',
    'GRU',
    'bidirectLSTM',
    'seq2seqLSTM',
    'stackedLSTM',
    'attentionLSTM',
    'TCN',
    'stackedTCN'
]
model_type = 'LSTM'

# Training defaults - rarely changed
blocks = 12                                 # number of RNN blocks
num_epochs = 50                             # number of training epochs
epochs_patience = 10                        # early stopping
batch_size = 1                              # faster training --> to be tested later
batch_size_test = 1                         # don't change

stacked_blocks = 3                          # 1 = no stack
batch_normalization = False                 # no significant effect when used with ADAM
dropout_rate = 1.0                          # range <0.5, 0.8>, 0.0=no outputs, 1.0=no dropout

# train_time_range = '<2019-04-15,2019-05-01)'   # 2 weeks
# train_time_range = '<2019-04-01,2019-05-01)'   # 1 month
# train_time_range = '<2019-02-01,2019-05-01)'   # 3 months
train_time_range = '<2018-11-01,2019-05-01)'     # 6 months - K20 experiments with k, p
# train_time_range = '<2018-08-01,2019-05-01)'   # 9 months
# train_time_range = '<2018-05-01,2019-05-01)'   # 1 year
#train_time_range_excluded = [
#    '<2018-12-01,2019-01-01)',
#    '<2019-01-01,2019-02-01)'
#]
train_time_range_excluded = []
train_ws_choices = ws_choices
train_ws = ws_choice

# prediction defaults
data_predict = 'sample-w1h-s10m.tsv'        # can be removed later?

# test defaults
test_data = 'data_test.tsv'                     # can be removed later? TODO: we first need to support datapool in the DEEPaaS web interface (@stevo)
test_data_select_query = data_select_query      # same as for train - differs only in the time range
# test_time_range = '<2019-05-01,2019-05-06)'   # paper plot - 5 days
test_time_range = '<2019-05-01,2019-06-01)'     # 1 month
# test_time_range = '<2019-05-01,2019-07-01)'   # 2 months for test
# test_time_range = <2019-05-01,2019-08-01)'    # 3 months

test_time_range_excluded = []
test_ws_choices = ws_choices
test_ws = ws_choice

# Data transformation defaults
interpolate = False

# common defaults
def list_models():
    return list_dir(app_models, '*.zip')

model_name = 'model-default.zip'
fill_missing_rows_in_timeseries = True                        # fills missing rows in time series data

# Evaluation metrics on real values
eval_metrics = ['SMAPE', 'R2', 'COSINE']    # 'MAPE', 'RMSE'
eval_filename = 'eval.tsv'

# Plotting
plot = False
plot_dir = os.path.join(BASE_DIR, 'plots')
plot_filename = 'plot_data.png'
fig_size_x = 25                             # max 2^16 pixels = 650 inch
fig_size_y = 4

# Auxiliary: DayTime format
format_string = '%Y-%m-%d %H:%M:%S'
format_string_parquet = '%Y-%m-%d %H_%M_%S'     # parquet format without ":"
timezone = 3600
