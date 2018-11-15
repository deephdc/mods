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
DATA_DIR = expanduser("~") + '/data/deep-dm/'           # app_data_raw
# Data dirs
dir_logs = DATA_DIR + 'logs/'
dir_parquet = DATA_DIR + 'logs_parquet/'
dir_cleaned = DATA_DIR + 'logs_cleaned/'
log_header_lines = 8


# Application dirs
app_data = BASE_DIR + '/data/'
app_data_raw = BASE_DIR + '/data/raw/'  # ln -s ...
app_data_features = BASE_DIR + '/data/features/'        # extracted features
app_models = BASE_DIR + '/models/'
app_checkpoints = BASE_DIR + '/checkpoints/'
app_visualization = BASE_DIR + '/visualization/'

default_model = 'serie_mock_1h_10m-seq-6.zip'

# Feature data
feature_filename = 'features.tsv'
# time_range_begin = '2018-04-14'         # begin <= time_range < end
# time_range_end   = '2018-10-15'         # excluded
time_range_begin = '2018-07-14'
time_range_end = '2018-10-15'
window_duration = '1 hour'
slide_duration = '10 minutes'

# ML data
column_separator = '\t'                                                     # for tsv
# column_separator = ','                                                    # for csv

# cols_included = ['Close']                                                 # yahoo_finance_stock.csv
# cols_included = ['number_of_conn', 'sum_orig_bytes', 'sum_resp_bytes']
cols_included = ['number_of_conn', 'sum_orig_bytes']
# cols_included = ['sum_orig_bytes', 'sum_resp_bytes']
# cols_included = ['number_of_conn']
# cols_included = ['sum_orig_bytes']


# ML and time series datasets
split_ratio = 0.67                      # train:test = 2:1
batch_size = 1                          # delta (6), without_delta(1)
sequence_len = 6                        # sequence lenght


# model properties
model_type  = 'LSTM'                    # 'LSTM', 'GRU', 'NN'
model_delta = True
blocks = 6
epochs = 50
epochs_patience = 10


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
