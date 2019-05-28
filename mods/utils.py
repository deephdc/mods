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

import datetime
import json
import os
import re
# from datetime import datetime
from math import sqrt

import keras
import numpy as np
import pandas as pd
from dateutil.relativedelta import *
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import hashlib

import mods.config as cfg


# matplotlib.style.use('ggplot')
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


# @giang: read .tsv file -> pandas dataframe -> numpy array
def read_data(data_filename):
    df = pd.read_csv(data_filename,
                     sep=cfg.pd_sep,
                     skiprows=0,
                     skipfooter=0,
                     engine='python',
                     usecols=lambda col: col in cfg.pd_usecols
                     )
    print(len(df.columns), list(df))

    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace('NaN', 0, inplace=True)
    df.interpolate(inplace=True)

    if 'sum_orig_kbytes' in list(df):
        df['sum_orig_kbytes'] = df['sum_orig_kbytes'].div(1024 * 1024).astype(int)
        # print(df)

    # Data: pandas dataframe to numpy array
    data = df.values.astype('float32')
    print('read_data: ', data_filename, data.dtype, cfg.multivariate)
    return data


# @giang: get X from TimeseriesGenerator data
def getX(tsg_data):
    X = list()
    for x, y in tsg_data:
        X.append(x)
    return np.array(X)


# @giang: get Y from TimeseriesGenerator data
def getY(tsg_data):
    Y = list()
    for x, y in tsg_data:
        Y.append(y)
    return np.array(Y)


# @giang: get XY from TimeseriesGenerator data
def getXY(tsg_data):
    X, Y = list(), list()
    for x, y in tsg_data:
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# @giang: first order differential d(y)/d(t)=f(y,t)=y' for numpy array
def delta_timeseries(arr):
    return arr[1:] - arr[:-1]


# @giang: RMSE for numpy array
def rmse(a, b):
    score = []
    for i in range(a.shape[1]):
        score.append(sqrt(mean_squared_error(a[:, i], b[:, i])))
    return score


# @giang: cosine similarity for two numpy arrays, <-1.0, 1.0>
def cosine(a, b):
    score = []
    for i in range(a.shape[1]):
        cos_sim = dot(a[:, i], b[:, i]) / (norm(a[:, i]) * norm(b[:, i]))
        score.append(cos_sim)
    return score


# @giang: R^2 (coefficient of determination) regression score, <-1.0, 1.0>, not a symmetric function
def r2(a, b):
    score = []
    for i in range(a.shape[1]):
        score.append(r2_score(a[:, i], b[:, i]))
    return score


# @giang/@stevo: MAPE = np.mean(np.abs((A-F)/A)) * 100
def mape(y_true, y_pred):
    assert isinstance(y_true, np.ndarray), 'numpy array expected for y_true in mape'
    assert isinstance(y_pred, np.ndarray), 'numpy array expected for y_pred in mape'
    score = []
    for i in range(y_true.shape[1]):
        try:
            s = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
            if np.isnan(s):
                s = str(s)
            score.append(s)
        except ZeroDivisionError:
            score.append(str(np.nan))
    return score


# @giang/@stevo: SMAPE = 100/len(A) * np.sum(2 * np.abs(F-A) / (np.abs(A) + np.abs(F))), symmetric function
def smape(y_true, y_pred):
    assert isinstance(y_true, np.ndarray), 'numpy array expected for y_true in smape'
    assert isinstance(y_pred, np.ndarray), 'numpy array expected for y_pred in smape'
    score = []
    for i in range(y_true.shape[1]):
        try:
            s = 100 / len(y_true[:, i]) * \
                np.sum(2 * np.abs(y_pred[:, i] - y_true[:, i]) / (np.abs(y_true[:, i]) + np.abs(y_pred[:, i])))
            if np.isnan(s):
                s = str(s)
            score.append(s)
        except ZeroDivisionError:
            score.append(str(np.nan))
    return score


##### @giang auxiliary - BEGIN - code in this block can be removed later #####
# @giang
def create_data_from_datapool(data_filename,
                              data_begin,
                              data_end,
                              data_excluded,
                              app_data_pool=cfg.app_data_pool,
                              app_data=cfg.app_data
                              ):
    flist = []
    for filename in sorted(os.listdir(app_data_pool)):
        if filename.endswith('.tsv'):
            fn = os.path.basename(filename).split('.')[0]
            if (data_begin <= fn <= data_end) and (fn not in data_excluded):
                flist.append(app_data_pool + filename)
    print(flist)

    filename = app_data + data_filename
    if not filename.endswith(('.tsv')):
        filename = filename + '.tsv'

    write_header = True
    with open(filename, 'w') as fout:
        for fn in flist:
            with open(fn) as fin:
                if write_header:
                    header = fin.readline()
                    fout.write(header)
                    write_header = False
                else:
                    next(fin)
                for line in fin:
                    fout.write(line)

    print('created data=', filename + '\n')
    return filename

##### @giang auxiliary - END - code in the above block can be removed later #####

##### @stevo @stevo @stevo#####

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


# @stevo
def parse_int_or_str(val):
    val = val.strip()
    try:
        return int(val)
    except Exception:
        return str(val)


# @stevo
def compute_metrics(y_true, y_pred, model):
    result = {}

    if len(y_true) > 1 and len(y_true) == len(y_pred):

        eval_result = model.eval(y_true)

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values

        err_mape = mape(y_true, y_pred)
        err_smape = smape(y_true, y_pred)
        err_r2 = r2(y_true, y_pred)
        err_rmse = rmse(y_true, y_pred)
        err_cosine = cosine(y_true, y_pred)

        result['mods_mape'] = err_mape
        result['mods_smape'] = err_smape
        result['mods_r2'] = err_r2
        result['mods_rmse'] = err_rmse
        result['mods_cosine'] = err_cosine

        i = 0
        for metric in model.model.metrics_names:
            result[metric] = eval_result[i]
            i += 1

    return result


# @stevo tsv representation of a dataframe
def df2tsv(df):
    if isinstance(df, pd.DataFrame):
        df = df.values
    ret = ''
    for row in df:
        for col in row:
            ret += str(col) + '\t'
        ret += '\n'
    return ret


# @stevo tsv representation of a tsg
def tsg2tsv(tsg):
    ret = ''
    for i in range(len(tsg)):
        x, y = tsg[i]
        ret += '%s => %s\n' % (x, y)
    return ret


# @stevo saves dataframe to a file
def save_df(df, model_name, file):
    dir = os.path.join(cfg.app_data, model_name[:-4] if model_name.lower().endswith('.zip') else model_name)
    if not os.path.isdir(dir):
        if os.path.isfile(dir):
            raise NotADirectoryError(dir)
        os.mkdir(dir)
    with open(os.path.join(dir, file), mode='w') as f:
        f.write(df2tsv(df))
        f.close()


# @stevo prints dataframe within a range of rows
def print_df(df, name, min=0, max=9):
    print('%s:\n%s' % (name, df2tsv(df[min:max])))


# @stevo prints dataframe to stdout and/or saves it to a file <<model>>/<<name>>.tsv
def dbg_df(df, model_name, df_name, print=False, save=False):
    if print:
        print_df(df, df_name)
    if save:
        save_df(df, model_name, df_name + '.tsv')


# @stevo prints TimeSeriesGenerator to stdout
def dbg_tsg(tsg, msg, debug=False):
    if debug:
        print('%s:\n%s' % (msg, tsg2tsv(tsg)))


# @stevo prints scaler to stdout
def dbg_scaler(scaler, msg, debug=False):
    if debug:
        print('%s - scaler.get_params(): %s\n\tscaler.data_min_=%s\n\tscaler.data_max_=%s\n\tscaler.data_range_=%s'
              % (
                  msg,
                  scaler.get_params(),
                  scaler.data_min_,
                  scaler.data_max_,
                  scaler.data_range_
              ))


# @stevo - parses data specification in order to support multiple data files merging
def parse_data_specs(specs):

    protocols = []
    merge_on_col = []

    specs = re.compile(r'\s*;\s*').split(specs.strip())

    if specs and len(specs) > 0:

        x = re.compile(r'\s*#\s*').split(specs[-1], 1)
        if x and len(x) == 2:
            specs[-1] = x[0]
            merge_on_col = list(filter(None, re.compile(r'\s*,\s*').split(x[1])))

        for spec in specs:
            # parse an array of file names (separated by |)
            parsed = re.compile(r'\s*\|\s*').split(spec)
            protocol = parsed[0]
            columns = parsed[1:] if len(parsed) > 1 else []

            # columns.extend(merge_on_col)
            protocols.append({'protocol': protocol, 'cols': columns})

    return (protocols, merge_on_col)


# @stevo
REGEX_DATAPOOLTIME = re.compile(r'^\s*(?P<year>\d{4})([^0-9]{0,1}(?P<month>\d{2})([^0-9]{0,1}(?P<day>\d{2}))?)?\s*$')
def parse_datetime(s):
    match = REGEX_DATAPOOLTIME.match(s)
    if match:
        y = int(match.group('year'))
        m = match.group('month')
        d = match.group('day')
        if m is None:
            return datetime.datetime(y, 1, 1)
        elif d is None:
            return datetime.datetime(y, int(m), 1)
        else:
            return datetime.datetime(y, int(m), int(d))
    return None


# @stevo
def expand_to_datetime(y, m, d, is_end=False, inclusive_end=cfg.time_range_inclusive):
    assert y is not None
    if m is None:
        if is_end and inclusive_end:
            return datetime.datetime(int(y), 12, 31)
        else:
            return datetime.datetime(int(y), 1, 1)
    elif d is None:
        if is_end and inclusive_end:
            return datetime.datetime(int(y), int(m), 1) + relativedelta(months=+1, days=-1)
        else:
            return datetime.datetime(int(y), int(m), 1)
    else:
        return datetime.datetime(int(y), int(m), int(d))


# @stevo
def expand_to_datetime_range(y, m, d, inclusive_end=cfg.time_range_inclusive):
    assert y is not None
    if m is None:
        if inclusive_end:
            d_beg = datetime.datetime(int(y), 1, 1)
            d_end = datetime.datetime(int(y), 12, 31)
            return (d_beg, d_end)
        else:
            d = datetime.datetime(int(y), 1, 1)
            return (d, d + relativedelta(years=+1))
    elif d is None:
        if inclusive_end:
            d_beg = datetime.datetime(int(y), int(m), 1)
            d_end = d_beg + relativedelta(months=+1, days=-1)
            return (d_beg, d_end)
        else:
            d = datetime.datetime(int(y), int(m), 1)
            return (d, d + relativedelta(months=+1))
    else:
        if inclusive_end:
            d = datetime.datetime(int(y), int(m), int(d))
            return (d, d)
        else:
            d = datetime.datetime(int(y), int(m), int(d))
            return (d, d + relativedelta(days=+1))


# @stevo
REGEX_DATAPOOLTIMERANGE = re.compile(r'^\s*(?P<beg_year>\d{4})([^0-9]{0,1}(?P<beg_month>\d{2})([^0-9]{0,1}(?P<beg_day>\d{2}))?)?\s*--\s*(?P<end_year>\d{4})([^0-9]{0,1}(?P<end_month>\d{2})([^0-9]{0,1}(?P<end_day>\d{2}))?)?\s*$')
def parse_datetime_ranges(time_ranges):
    parsed = []
    if isinstance(time_ranges, str):
        time_ranges = re.compile(r'\s*,\s*').split(time_ranges)
    if time_ranges is None:
        return parsed
    for x in time_ranges:
        m = REGEX_DATAPOOLTIME.match(x)
        if m:
            print('matched datetime: %s' % x)
            # single date specified; e.g. 2019, 2019-01, 2019-01-01
            r = expand_to_datetime_range(
                m.group('year'),
                m.group('month'),
                m.group('day')
            )
            parsed.append(r)
            continue
        m = REGEX_DATAPOOLTIMERANGE.match(x)
        print(m)
        if m:
            print('matched datetime_range: %s' % x)
            beg = expand_to_datetime(
                m.group('beg_year'),
                m.group('beg_month'),
                m.group('beg_day')
            )
            end = expand_to_datetime(
                m.group('end_year'),
                m.group('end_month'),
                m.group('end_day'),
                is_end = True
            )
            parsed.append((beg, end))
            continue
    return parsed


# @stevo
def is_within_range(d, range, inclusive_end=cfg.time_range_inclusive):
    if inclusive_end:
        return range[0] <= d and d <= range[1]
    else:
        return range[0] <= d and d < range[1]


# @stevo
def exclude(d, ranges):
    for range in ranges:
        if is_within_range(d, range):
            return True


# @stevo
# regex matching directory of a day
REGEX_DIR_DAY = re.compile(r'^' + re.escape(cfg.app_data_features.rstrip('/')) + '/[^/]+' + r'/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})')


# @stevo datapool reading
def datapool_read(
        data_specs_str,                 # protocol/column/merge specification
        time_range,                     # (beg datetime.datetime, end datetime.datetime)
        ws,                             # window/slide specification; e.g., w01h-s10m
        excluded=[],                    # list of dates and ranges that will be omitted
        base_dir=cfg.app_data_features  # base dir with the protocol/YYYY/MM/DD/wXXd-sXXd.tsv structure
):
    keep_cols = []
    df_main = None

    protocols, merge_on_col = parse_data_specs(data_specs_str)

    # read dataset from cache
    cache_dir = None
    cache_key = None
    cache_file = None
    if cfg.data_pool_caching:
        cache_dir = os.path.dirname(cfg.app_data_pool_cache)
        cache_key = data_cache_key(protocols, merge_on_col, ws, time_range, excluded)
        cache_file = os.path.join(cache_dir, cache_key)
        if os.path.isfile(cache_file):
            df = pd.read_csv(
                cache_file,
                header=0,
                sep='\t',
                skiprows=0,
                skipfooter=0,
                engine='python',
            )
            return df, cache_file

    for ds in protocols:

        dir_protocol = os.path.join(base_dir, ds['protocol'])

        # protocol's collecting df
        df_protocol = None

        # collect columns, that will be kept in the final dataset
        keep_cols.extend(ds['cols'])

        # columns to be loaded: columns specified for the file as well as columns, that will be used for joins
        ds['cols'].extend(merge_on_col)

        for root, directories, filenames in os.walk(dir_protocol):

            # *.tsv base dir filter
            rematch = REGEX_DIR_DAY.match(root)
            if not rematch:
                continue

            for f in filenames:

                # windows/slide filter
                if not f.startswith(ws):
                    continue

                # exclusion filter
                dpt = datetime.datetime(
                    int(rematch.group('year')),
                    int(rematch.group('month')),
                    int(rematch.group('day'))
                )

                data_file = os.path.join(root, f)
                if exclude(dpt, excluded) or not is_within_range(dpt, time_range):
                    print('skipping: %s' % data_file)
                    continue

                # load one of the data files
                print('loading: %s' % data_file)
                df = pd.read_csv(
                    open(data_file),
                    usecols=ds['cols'],
                    header=0,
                    sep='\t',
                    skiprows=0,
                    skipfooter=0,
                    engine='python',
                )

                if df_protocol is None:
                    df_protocol = df
                else:
                    df_protocol = df_protocol.append(df)

        if df_protocol is None:
            continue

        if df_main is None:
            df_main = df_protocol
        else:
            df_main = pd.merge(df_main, df_protocol, on=merge_on_col)

    # select only specified columns
    df_main = df_main[keep_cols]

    # save dataset to cache
    if cfg.data_pool_caching:
        assert cache_dir is not None
        assert cache_key is not None
        assert cache_file is not None
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=False)
        df_main.to_csv(
            cache_file,
            index=None,
            header=True,
            sep='\t'
        )

    return df_main, cache_file


# @stevo - converts datetime.datetime dates to str in order to overcome json serialization error.
def datetime2str(obj):
    if isinstance(obj, datetime.datetime):
        return obj.strftime('%Y-%m-%d')
    if isinstance(obj, list) or isinstance(obj, tuple):
        a = []
        for x in obj:
            a.append(datetime2str(x))
        return a
    return obj


# @stevo - helper function for protocol dict comparison
def compare_protocol_spec(p):
    assert isinstance(p, dict)
    return (str(p['protocol']) + str(sorted(p['cols']))).lower()


# @stevo - computes hash key for caching
def data_cache_key(protocols, merge_on_col, ws, time_range, excluded):
    protocols = sorted(protocols, key=compare_protocol_spec)
    merge_on_col = sorted(merge_on_col)
    k = str(str(protocols) + str(merge_on_col) + ';' + ws + ';' + str(datetime2str(time_range)) + ';' + str(datetime2str(excluded))).lower()
    print('HASH in: %s' % k)
    m = hashlib.md5()
    m.update(k.encode('utf-8'))
    return m.hexdigest()