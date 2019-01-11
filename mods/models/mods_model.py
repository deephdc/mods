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
Created on Mon Jan 11 13:34:37 2019

@author: giang nguyen
@author: stefan dlugolinsky
"""

import io
import json
import os
import tempfile
from zipfile import ZipFile

import keras
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

# import project config.py
import mods.config as cfg


class mods_model:
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
            self.__load_model(zip, self.config[mods_model.__MODEL])
            self.__load_scaler(zip, self.config[mods_model.__SCALER])
            self.__load_sample_data(zip, self.config[mods_model.__SAMPLE_DATA])
        print('Model loaded')

    def __load_config(self, zip, file):
        print('Loading model config')
        with zip.open(file) as f:
            data = f.read()
            self.config = json.loads(data.decode('utf-8'))
        print('Model config:\n%s' % json.dumps(self.config, indent=True))

    def __load_model(self, zip, config):
        print('Loading keras model')
        with zip.open(config[mods_model.__FILE]) as f:
            self.model = self.__func_over_tempfile(f, keras.models.load_model)
        print('Keras model loaded')

    def __load_scaler(self, zip, config):
        print('Loading scaler')
        with zip.open(config[mods_model.__FILE]) as f:
            self.scaler = joblib.load(f)
        print('Scaler loaded')

    def __load_sample_data(self, zip, config):
        print('Loading sample data')
        with zip.open(config[mods_model.__FILE]) as f:
            self.sample_data = pd.read_csv(io.TextIOWrapper(f),
                                           sep=config[mods_model.__SEP],
                                           skiprows=config[mods_model.__SKIPROWS],
                                           skipfooter=config[mods_model.__SKIPFOOTER],
                                           engine=config[mods_model.__ENGINE],
                                           usecols=lambda col: col in config[mods_model.__USECOLS]
                                           )
        print('Sample data loaded:\n%s' % self.sample_data)

    def __default_config(self):
        return {
            mods_model.__MODEL: {
                mods_model.__FILE: 'model.h5',
                mods_model.__MULTIVARIATE: len(cfg.cols_included),
                mods_model.__SEQUENCE_LEN: cfg.sequence_len,
                mods_model.__MODEL_DELTA: cfg.model_delta
            },
            mods_model.__SCALER: {
                mods_model.__FILE: 'scaler.pkl'
            },
            mods_model.__SAMPLE_DATA: {
                mods_model.__FILE: 'sample_data.tsv',
                mods_model.__SEP: '\t',
                mods_model.__SKIPROWS: 0,
                mods_model.__SKIPFOOTER: 0,
                mods_model.__ENGINE: 'python',
                mods_model.__USECOLS: lambda col: col in cfg.cols_included
            }
        }

    def cfg_model(self):
        return self.config[mods_model.__MODEL]

    def set_multivariate(self, multivariate):
        self.cfg_model()[mods_model.__MULTIVARIATE] = multivariate

    def get_multivariate(self):
        return self.cfg_model()[mods_model.__MULTIVARIATE]

    def set_sequence_len(self, sequence_len):
        self.cfg_model()[mods_model.__SEQUENCE_LEN] = sequence_len

    def get_sequence_len(self):
        return self.cfg_model()[mods_model.__SEQUENCE_LEN]

    def set_model_delta(self, model_delta):
        self.cfg_model()[mods_model.__MODEL_DELTA] = model_delta

    def isdelta(self):
        return self.cfg_model()[mods_model.__MODEL_DELTA]

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
        if model_type == 'GRU':
            h = GRU(cfg.blocks)(x)
        elif model_type == 'bidirect':
            h = Bidirectional(LSTM(cfg.blocks))(x)
        elif model_type == 'seq2seq':
            h = LSTM(cfg.blocks)(x)
            h = RepeatVector(sequence_len)(h)
            h = LSTM(cfg.blocks, return_sequences=True)(h)
            h = Flatten()(h)
        elif model_type == 'CNN':
            h = Conv1D(filters=64, kernel_size=2, activation='relu')(x)
            h = MaxPooling1D(pool_size=2)(h)
            h = Flatten()(h)
        elif model_type == 'MLP':
            h = Dense(units=multivariate, activation='relu')(x)
            # h = Dense(units=multivariate, activation='relu')(h)
            h = Flatten()(h)
        else:  # default LSTM
            h = LSTM(cfg.blocks)(x)
            # h = LSTM(cfg.blocks)(h)         # stacked
        y = Dense(units=multivariate, activation='sigmoid')(h)  # 'softmax'
        self.model = Model(inputs=x, outputs=y)

        # Drawing model
        print(self.model.summary())

        # Compile model
        self.model.compile(loss='mean_squared_error',
                           optimizer='adam',  # 'adagrad', 'rmsprop'
                           metrics=['mse', 'mae', 'mape'])  # 'cosine'

        # Checkpointing and earlystopping
        filepath = cfg.app_checkpoints + self.name + '-{epoch:02d}.hdf5'
        checkpoints = ModelCheckpoint(filepath, monitor='loss',
                                      save_best_only=True,
                                      mode=max,
                                      verbose=1
                                      )
        earlystops = EarlyStopping(monitor='loss',
                                   patience=cfg.epochs_patience,
                                   verbose=1
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
