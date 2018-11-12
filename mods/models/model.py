# -*- coding: utf-8 -*-
"""
Model description
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

import numpy as np
import pandas as pd

import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Input
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


class MODSModel:
    # generic
    __FILE = 'file'
    # model
    __MODEL = 'model'
    __MULTIVARIATE = 'multivariate'
    __SEQUENCE_LEN = 'sequence_len'
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

    def set_sequence_len(self, sequence_len):
        self.cfg_model()[MODSModel.__SEQUENCE_LEN] = sequence_len

    def get_multivariate(self):
        return self.cfg_model()[MODSModel.__MULTIVARIATE]

    def get_sequence_len(self):
        return self.cfg_model()[MODSModel.__SEQUENCE_LEN]

    def create_model(self,
                     multivariate=None,
                     sequence_len=None,
                     use_GRU=cfg.use_GRU,  # todo: use_GRU --> model_type; e.g., NN, LSTM, GRU
                     blocks=cfg.blocks):

        if multivariate:
            self.set_multivariate(multivariate)
        else:
            multivariate = self.get_multivariate()
        if sequence_len:
            self.set_sequence_len(sequence_len)
        else:
            sequence_len = self.get_sequence_len()

        # Define model
        x = Input(shape=(sequence_len, multivariate))
        try:
            if use_GRU:
                h = GRU(blocks)(x)
            else:
                h = LSTM(blocks)(x)
        except Exception as e:
            print(e)
            h = LSTM(blocks)(x)

        y = Dense(multivariate, activation='sigmoid')(h)  # 'sigmoid', 'softmax'
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

    def transform(self, df):
        # First order differential for numpy array      y' = d(y)/d(t) = f(y,t)
        # be carefull                                   len(dt) == len(data)-1
        # todo: move here from the utils.py
        df = utl.delta_timeseries(df)
        return df

    # normalizes data
    def normalize(self, df):
        # Scale all metrics but each separately
        df = self.scaler.fit_transform(df)
        return df

    # denormalizes time series
    def denormalize(self, tsg):
        return self.scaler.inverse_transform(tsg)

    # returns time series generator
    def get_tsg(self, df):
        return TimeseriesGenerator(df,
                                   df,
                                   length=self.get_sequence_len(),
                                   sampling_rate=1,
                                   stride=1,
                                   batch_size=1)

    def predict(self, df):
        df = df.interpolate()
        df = df.values.astype('float32')
        df = self.transform(df)
        df = self.normalize(df)
        tsg = self.get_tsg(df)
        prediction = self.model.predict_generator(tsg)
        df = self.denormalize(prediction)
        return df


# load model
model_filename = os.path.join(cfg.app_models, cfg.default_model)
mods_model = MODSModel(model_filename)

if not mods_model:
    print('Could not load model: %s' % model_filename)
    sys.exit(1)


# def get_sample_data():
#     return sample_data()
#     # df = pd.DataFrame(sample_data)
#     # df.interpolate(inplace=True)
#     # df = df.values.astype('float32')
#     # df = transform(df)
#     # return df


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
            message['predictions'] = df.to_json()
    return message


def predict_url(*args):
    """
    Function to make prediction on a URL
    """
    message = 'Not implemented in the model (predict_url)'
    return message


def train(*args):
    """
    Train network
    """
    message = 'Not implemented in the model (train)'
    return message
