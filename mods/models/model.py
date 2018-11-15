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
Created on Mon Oct 15 10:14:37 2018

Train models with first order differential to monitor changes

@author: giang nguyen
@author: stefan dlugolinsky
"""

import pkg_resources
# import project config.py
import mods.config as cfg
# import utilities
import mods.utils as utl

import io

import os

import sys
import json
import tempfile

from zipfile import ZipFile

import pandas as pd

import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

import socket


class MODSModel:
    # generic
    __FILE = 'file'
    # model
    __MODEL = 'model'
    __MULTIVARIATE = 'multivariate'
    __SEQUENCE_LEN = 'sequence_len'
    __MODEL_DELTA = 'model_delta'
    # scaler
    __SCALER = 'scaler'
    # sample data
    __SAMPLE_DATA = 'sample_data'
    __SEP = 'sep'
    __SKIPROWS = 'skiprows'
    __SKIPFOOTER = 'skipfooter'
    __ENGINE = 'engine'
    __USECOLS = 'usecols'

    def __init__(self, file):
        self.file = file
        self.name = os.path.basename(file)
        self.config = None
        self.model = None
        self.scaler = None
        self.sample_data = None
        if os.path.isfile(file):
            self.__load(file)
        else:
            self.config = self.__default_config()
            self.model = self.create_model()
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.__init()

    # saves the contents of the original file (e.g. file in a zip) into a temp file and runs func over it
    def __func_over_tempfile(self, orig_file, func, mode='wb', *args, **kwargs):
        # create temp file
        _, fname = tempfile.mkstemp()
        with open(fname, mode) as tf:
            # extract model to the temp file
            tf.write(orig_file.read())
            # call the func over the temp file
            result = func(fname, *args, **kwargs)
        # remove the temp file
        os.remove(fname)
        return result

    def __load(self, file):
        print('Loading model: %s' % file)
        with ZipFile(file) as zip:
            self.__load_config(zip, 'config.json')
            self.__load_model(zip, self.config[MODSModel.__MODEL])
            self.__load_scaler(zip, self.config[MODSModel.__SCALER])
            self.__load_sample_data(zip, self.config[MODSModel.__SAMPLE_DATA])
        print('Model loaded')

    def __load_config(self, zip, file):
        print('Loading model config')
        with zip.open(file) as f:
            self.config = json.load(f)
        print('Model config:\n%s' % json.dumps(self.config, indent=True))

    def __load_model(self, zip, config):
        print('Loading keras model')
        with zip.open(config[MODSModel.__FILE]) as f:
            self.model = self.__func_over_tempfile(f, keras.models.load_model)
        print('Keras model loaded')

    def __load_scaler(self, zip, config):
        print('Loading scaler')
        with zip.open(config[MODSModel.__FILE]) as f:
            self.scaler = joblib.load(f)
        print('Scaler loaded')

    def __load_sample_data(self, zip, config):
        print('Loading sample data')
        with zip.open(config[MODSModel.__FILE]) as f:
            self.sample_data = pd.read_csv(io.TextIOWrapper(f),
                                           sep=config[MODSModel.__SEP],
                                           skiprows=config[MODSModel.__SKIPROWS],
                                           skipfooter=config[MODSModel.__SKIPFOOTER],
                                           engine=config[MODSModel.__ENGINE],
                                           usecols=lambda col: col in config[MODSModel.__USECOLS]
                                           )
        print('Sample data loaded:\n%s' % self.sample_data)

    def __default_config(self):
        return {
            MODSModel.__MODEL: {
                MODSModel.__FILE: 'model.h5',
                MODSModel.__MULTIVARIATE: len(cfg.cols_included),
                MODSModel.__SEQUENCE_LEN: cfg.sequence_len,
                MODSModel.__MODEL_DELTA: cfg.model_delta
            },
            MODSModel.__SCALER: {
                MODSModel.__FILE: 'scaler.pkl'
            },
            MODSModel.__SAMPLE_DATA: {
                MODSModel.__FILE: 'sample_data.tsv',
                MODSModel.__SEP: '\t',
                MODSModel.__SKIPROWS: 0,
                MODSModel.__SKIPFOOTER: 0,
                MODSModel.__ENGINE: 'python',
                MODSModel.__USECOLS: lambda col: col in cfg.cols_included
            }
        }

    def cfg_model(self):
        return self.config[MODSModel.__MODEL]

    def set_multivariate(self, multivariate):
        self.cfg_model()[MODSModel.__MULTIVARIATE] = multivariate

    def get_multivariate(self):
        return self.cfg_model()[MODSModel.__MULTIVARIATE]

    def set_sequence_len(self, sequence_len):
        self.cfg_model()[MODSModel.__SEQUENCE_LEN] = sequence_len

    def get_sequence_len(self):
        return self.cfg_model()[MODSModel.__SEQUENCE_LEN]

    def set_model_delta(self, model_delta):
        self.cfg_model()[MODSModel.__MODEL_DELTA] = model_delta

    def isdelta(self):
        return self.cfg_model()[MODSModel.__MODEL_DELTA]

    def create_model(self,
                     multivariate=None,
                     sequence_len=None,
                     model_type=cfg.model_type,
                     model_delta=cfg.model_delta,
                     blocks=cfg.blocks):

        if multivariate:
            self.set_multivariate(multivariate)
        else:
            multivariate = self.get_multivariate()
        if sequence_len:
            self.set_sequence_len(sequence_len)
        else:
            sequence_len = self.get_sequence_len()
        if model_delta:
            self.set_model_delta(model_delta)

        # Define model
        x = Input(shape=(sequence_len, multivariate))
        if model_type == 'NN':
            h = Dense(units=multivariate, activation='relu')(x)
            h = Flatten()(h)
        elif model_type == 'GRU':
            h = GRU(blocks)(x)
        else:
            h = LSTM(blocks)(x)
        y = Dense(units=multivariate, activation='sigmoid')(h)  # 'sigmoid', 'softmax'
        self.model = Model(inputs=x, outputs=y)

        # Drawing model
        print(self.model.summary())

        # Compile model
        self.model.compile(loss='mean_squared_error',
                           optimizer='adam',  # 'adagrad', 'rmsprop'
                           metrics=['mse', 'mae'])  # 'mape', 'cosine'

        # Checkpointing and earlystopping
        filepath = cfg.app_checkpoints + self.name + '-{epoch:02d}.hdf5'
        checkpoints = ModelCheckpoint(filepath, monitor='loss',
                                      save_best_only=True, mode=max, verbose=1
                                      )
        earlystops = EarlyStopping(monitor='loss',
                                   patience=cfg.epochs_patience, verbose=1
                                   )
        callbacks_list = [checkpoints, earlystops]

    def __init(self):
        print('Initializing model')
        self.predict(self.sample_data)
        print('Model initialized')

    # First order differential for numpy array      y' = d(y)/d(t) = f(y,t)
    # be carefull                                   len(dt) == len(data)-1
    # e.g., [5,2,9,1] --> [2-5,9-2,1-9] == [-3,7,-8]
    def delta(self, df):
        return df[1:] - df[:-1]

    def transform(self, df):
        if self.isdelta():
            return self.delta(df)
        else:
            # bucketing, taxo, fuzzy
            return df

    def inverse_transform(self, original, transformed, prediction):
        if self.isdelta():
            beg = self.get_sequence_len()
            end = beg + len(prediction)
            y = original[beg + 1:end + 1]
            return y - transformed[beg:end] + prediction
        else:
            return prediction

    # normalizes data
    def normalize(self, df):
        # Scale all metrics but each separately
        df = self.scaler.fit_transform(df)
        return df

    # inverse method to @normalize
    def inverse_normalize(self, df):
        return self.scaler.inverse_transform(df)

    # returns time series generator
    def get_tsg(self, df):
        return TimeseriesGenerator(df,
                                   df,
                                   length=self.get_sequence_len(),
                                   sampling_rate=1,
                                   stride=1,
                                   batch_size=1)

    def predict(self, df):

        interpol = df.interpolate()
        interpol = interpol.values.astype('float32')
        # print('interpolated:\n%s' % interpol)

        trans = self.transform(interpol)
        # print('transformed:\n%s' % transf)

        norm = self.normalize(trans)
        # print('normalized:\n%s' % norm)

        tsg = self.get_tsg(norm)
        pred = self.model.predict_generator(tsg)
        # print('prediction:\n%s' % pred)

        denorm = self.inverse_normalize(pred)
        # print('denormalized:\n%s' % denorm)

        invtrans = self.inverse_transform(interpol, trans, denorm)
        # print('inverse transformed:\n%s' % invtrans)

        return invtrans


# load model
model_filename = os.path.join(cfg.app_models, cfg.default_model)
mods_model = MODSModel(model_filename)

if not mods_model:
    print('Could not load model: %s' % model_filename)
    sys.exit(1)


def get_metadata():
    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': 'MODS - Massive Online Data Streams',
        'Version': '0.1',
        'Summary': 'Intelligent module using ML/DL techniques for underlying IDS and monitoring system',
        'Home-page': None,
        'Author': 'Giang Nguyen, Stefan Dlugolinsky',
        'Author-email': 'giang.nguyen@savba.sk, stefan.dlugolinsky@savba.sk',
        'License': 'Apache-2',
    }

    for l in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if l.startswith(par):
                _, v = l.split(": ", 1)
                meta[par] = v

    return meta


def predict_file(*args):
    """
    Function to make prediction on a local file
    """
    message = 'Not implemented in the model (predict_file)'
    return message


def predict_data(*args):
    """
    Function to make prediction on an uploaded file
    """
    message = 'Error reading input data'
    if args:
        for data in args:
            message = {'status': 'ok', 'predictions': []}
            df = pd.read_csv(io.BytesIO(data[0]), sep='\t', skiprows=0, skipfooter=0, engine='python')
            predictions = mods_model.predict(df)
            message['predictions'] = predictions.tolist()
    return message


def predict_stream(*args):
    """
    Function to make prediction on a stream

    Sample *args
    {
        "host": "127.0.0.1",
        "port": 9999,
        "columns": [
            "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service", "duration",
            "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
            "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents"
        ]
    }
    """
    print('args: %s' % args)

    conf = json.loads(args[0])
    host = conf['host']
    port = int(conf['port'])

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print('connecting to %s:%s' % (host, port))
        sock.connect((host, port))
        print('success')
    except Exception as e:
        message = str(e)
        return message

    chunks = []
    chunks_to_join = 5
    chunks_collected = 0
    chunk_len = 128
    data = b''

    seq_len = mods_model.get_sequence_len()

    while True:
        chunk = sock.recv(chunk_len)
        print('chunk: %s' % chunk)
        if chunk == b'':
            print('End of stream')
            break
        chunks.append(chunk)
        chunks_collected += 1
        if chunks_collected == chunks_to_join:
            tmp = b''.join(chunks)
            last = tmp.rfind(b'\n')
            # completed lines
            data = tmp[:last + 1]
            # incomplete line
            chunks = [tmp[last + 1:]]
            chunks_collected = 0
            print(data)
            df = pd.read_csv(io.BytesIO(data), sep='\t', skiprows=0, skipfooter=0, engine='python', header=None, dtype=str)
            print(df)
            predictions = mods_model.predict(df)
            message = {'status': 'ok', 'predictions': predictions.tolist()}
            print(message)


def train(*args):
    """
    Train network
    """
    message = 'Not implemented in the model (train)'
    return message
