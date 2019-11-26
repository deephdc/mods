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
import hashlib
import os
import re
from math import sqrt

import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import *
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import mods.config as cfg
from mods.mods_types import TimeRange


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


# @giang: read .tsv file -> pandas dataframe
def create_df(filename):
    df = pd.read_csv(filename,
                     sep=cfg.pd_sep,
                     skiprows=0,
                     skipfooter=0,
                     engine='python'
                     # usecols=lambda col: col in cfg.pd_usecols
                     )
    # data cleaning + missing values
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace(['NaN', np.nan], 0, inplace=True)

    # Intermittent Demand Analysis (IDA) or Sparse Data Analysis (SDA)
    # df.interpolate(inplace=True)

    # in_sum_orig_bytes, in_sum_resp_bytes, out_sum_orig_bytes, out_sum_resp_bytes in MB or GB
    for feature in list(df):
        if '_bytes' in feature:
            df[feature] = df[feature].div(1024 * 1024).astype(int)

    print('create_df', filename, '\t', len(df.columns), df.shape, '\n', list(df))
    return df


# @giang: read .tsv file -> pandas dataframe -> numpy array
def read_data(filename):
    df = create_df(filename)

    # Data: pandas dataframe to numpy array
    data = df.values.astype('float32')

    print('read_data: ', filename, '\t', data.shape[1], data.dtype, '\n', list(df))
    return data


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


##### @stevo @stevo @stevo#####


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

        # err_mape = mape(y_true, y_pred)
        err_smape = smape(y_true, y_pred)
        err_r2 = r2(y_true, y_pred)
        err_rmse = rmse(y_true, y_pred)
        err_cosine = cosine(y_true, y_pred)

        # result['mods_mape'] = err_mape
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
REGEX_SPLIT_SEMICOLON = re.compile(r'\s*;\s*')
REGEX_SPLIT_COMMA = re.compile(r'\s*,\s*')
REGEX_SPLIT_HASH = re.compile(r'\s*#\s*')
REGEX_SPLIT_PIPE = re.compile(r'\s*\|\s*')
REGEX_SPLIT_TILDE = re.compile(r'\s*~\s*')


def parse_data_specs(specs):
    protocols = []
    merge_on_col = []

    specs = REGEX_SPLIT_SEMICOLON.split(specs.strip())

    if specs and len(specs) > 0:

        x = REGEX_SPLIT_HASH.split(specs[-1], 1)
        if x and len(x) == 2:
            specs[-1] = x[0]
            merge_on_col = list(filter(None, REGEX_SPLIT_COMMA.split(x[1])))

        for spec in specs:
            # parse an array of file names (separated by |)
            parsed = REGEX_SPLIT_PIPE.split(spec)
            protocol = parsed[0]
            columns = parsed[1:] if len(parsed) > 1 else []
            # column rename rules
            columns = [REGEX_SPLIT_TILDE.split(col, 1) for col in columns]
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
REGEX_DATAPOOLTIMERANGE = re.compile(
    r'^\s*(?P<beg_year>\d{4})([^0-9]{0,1}(?P<beg_month>\d{2})([^0-9]{0,1}(?P<beg_day>\d{2}))?)?\s*--\s*(?P<end_year>\d{4})([^0-9]{0,1}(?P<end_month>\d{2})([^0-9]{0,1}(?P<end_day>\d{2}))?)?\s*$')


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
                is_end=True
            )
            parsed.append((beg, end))
            continue
    return parsed


# @stevo
def is_within_range(d, range: TimeRange):
    if range.is_lclosed():
        if range.is_rclosed():
            return range.beg <= d and d <= range.end
        else:
            return range.beg <= d and d < range.end
    else:
        if range.is_rclosed():
            return range.beg < d and d <= range.end
        else:
            return range.beg < d and d < range.end


# @stevo
def exclude(d, ranges):
    for range in ranges:
        if is_within_range(d, range):
            return True


# @stevo datapool reading
def datapool_read(
        data_specs_str,  # protocol/column/merge specification
        time_range,  # (beg datetime.datetime, end datetime.datetime)
        ws,  # window/slide specification; e.g., w01h-s10m
        excluded=[],  # list of dates and ranges that will be omitted
        base_dir=cfg.app_data_features,  # base dir with the protocol/YYYY/MM/DD/wXXd-sXXd.tsv structure
        caching=cfg.data_pool_caching
):
    # regex matching directory of a day
    REGEX_DIR_DAY = re.compile(r'^' + re.escape(
        base_dir.rstrip('/')) + '/[^/]+' + r'/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})')

    keep_cols = []
    df_main = None

    protocols, merge_on_col = parse_data_specs(data_specs_str)

    # read dataset from cache
    cache_dir = None
    cache_key = None
    cache_file = None
    if caching:
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

        # original column names
        cols_orig = [x[0] for x in ds['cols']]
        # column names after renaming
        cols = [x[1] if len(x) == 2 else x[0] for x in ds['cols']]

        # collect columns, that will be kept in the final dataset
        keep_cols.extend(cols)

        # columns to be loaded: columns specified for the file as well as columns, that will be used for joins
        # TODO: check duplicate columns?
        cols_orig.extend(merge_on_col)

        for root, directories, filenames in os.walk(dir_protocol):

            # *.tsv base dir filter
            rematch = REGEX_DIR_DAY.match(root)
            if not rematch:
                continue

            for f in filenames:

                # windows/slide filter
                if not f.startswith(ws):
                    continue

                year = int(rematch.group('year'))
                month = int(rematch.group('month'))
                day = int(rematch.group('day'))

                # exclusion filter
                dpt = datetime.datetime(year, month, day, tzinfo=pytz.UTC)

                data_file = os.path.join(root, f)
                if exclude(dpt, excluded) or not is_within_range(dpt, time_range):
                    print('skipping: %s' % data_file)
                    continue

                # load one of the data files
                print('loading: %s' % data_file)
                fp = open(data_file)
                df = pd.read_csv(
                    fp,
                    usecols=cols_orig,
                    header=0,
                    sep='\t',
                    skiprows=0,
                    skipfooter=0,
                    engine='python',
                )
                fp.close()

                if cfg.fill_missing_rows_in_timeseries:
                    # fill missing rows for the loaded day
                    range_beg = '%d-%02d-%02d' % (year, month, day)
                    range_end = str(expand_to_datetime(year, month, day) + relativedelta(days=+1))
                    df = fill_missing_rows(
                        df,
                        range_beg=range_beg,
                        range_end=range_end
                    )

                if df_protocol is None:
                    df_protocol = df
                else:
                    df_protocol = df_protocol.append(df)

        if df_protocol is None:
            continue

        # rename columns
        rename_rule = {x[0]: x[1] for x in ds['cols'] if len(x) == 2}
        df_protocol = df_protocol.rename(index=str, columns=rename_rule)

        # convert units:
        # from B to kB, MB, GB use _kB, MB, GB
        for col in df_protocol.columns:
            if col.lower().endswith('_kb'):
                df_protocol[col] = df_protocol[col].div(1024).astype(int)
            elif col.lower().endswith('_mb'):
                df_protocol[col] = df_protocol[col].div(1048576).astype(int)
            elif col.lower().endswith('_gb'):
                df_protocol[col] = df_protocol[col].div(1073741824).astype(int)

        if df_main is None:
            df_main = df_protocol
        else:
            df_main = pd.merge(df_main, df_protocol, on=merge_on_col)

    # select only specified columns
    df_main = df_main[keep_cols]

    # save dataset to cache
    if caching:
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
    key_str = str(
        str(protocols) \
        + str(merge_on_col) \
        + ';' \
        + ws \
        + ';' \
        + str(datetime2str(time_range)) \
        + ';' \
        + str(datetime2str(excluded))
    ).lower()
    m = hashlib.md5()
    m.update(key_str.encode('utf-8'))
    return m.hexdigest()


# @stevo
def fix_missing_num_values(df, cols=None):
    if cols:
        for col in cols:
            df['col'] = pd.to_numeric(df['col'], errors='coerce')
            df['col'] = df['col'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
            df['col'] = df['col'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['col'] = df['col'].replace(['NaN', np.nan], 0, inplace=True)
            df['col'] = df['col'].interpolate(inplace=True)
    else:
        df = df.apply(pd.to_numeric, errors='coerce')
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.replace(['NaN', np.nan], 0, inplace=True)
        df.interpolate(inplace=True)
    return df


# @stevo
def estimate_window_spec(df):
    tmpdf = df[['window_start', 'window_end']]
    slide_duration = tmpdf['window_start'].diff()[1:].min()
    row1 = tmpdf[:1]
    window_duration = row1.window_end[:1].iloc[0] - row1.window_start[:1].iloc[0]
    return window_duration, slide_duration


# @stevo
def fill_missing_rows(df, range_beg=None, range_end=None):
    """Fills the missing rows in the time series dataframe by estimating the slide and window duration.

    Parameters
    ----------
    df : pandas.DataFrame
        Data
    range_beg : str
        Time range begin in 'YYYY-MM-DD hh:mm:ss' format (default is None)
    range_end : str
        Time range end in 'YYYY-MM-DD hh:mm:ss' format (default is None)

    Returns
    -------
    pandas.DataFrame
        DataFrame filled with missing rows
    """
    if not ('window_start' in df.columns and 'window_end'):
        return df
    numrows = len(df.index)
    # convert cols to datetime
    df = df.apply(lambda x: pd.to_datetime(x) if x.name in ['window_start', 'window_end'] else x)
    # estimate window specification
    window_duration, slide_duration = estimate_window_spec(df)
    tz = df[:1]['window_start'].iloc[0].tzinfo
    if range_beg:
        range_beg = pd.Timestamp(range_beg, tzinfo=tz)
        if range_beg < df[:1]['window_start'].iloc[0]:
            # add the first row for the specified range to fill from
            df = df.shift()
            df.loc[0, 'window_start'] = range_beg
            df.loc[0, 'window_end'] = range_beg + window_duration
    if range_end:
        range_end = pd.Timestamp(range_end, tzinfo=tz)
        if range_end > df[-1:]['window_end'].iloc[0]:
            # add the last row for the specified range to fill to
            df = df.append(pd.Series(), ignore_index=True)
            df.loc[df.index[-1], 'window_start'] = range_end - window_duration
            df.loc[df.index[-1], 'window_end'] = range_end
    # set df index
    df = df.set_index('window_start')
    # fill missing rows using slide_duration as the frequency
    df = df.asfreq(slide_duration)
    # reset index to use window_start as a column
    df = df.reset_index(level=0)
    # compute window_end values for the newly added rows (not necessary at the moment)
    df['window_end'] = df['window_start'] + window_duration
    newnumrows = len(df.index)
    if newnumrows > numrows:
        print('filled %d missing rows (was %d)' % (newnumrows - numrows, numrows))
    return df

# @stevo
