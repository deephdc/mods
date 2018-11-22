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

import numpy as np
import pandas as pd

import keras
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

import socket
from threading import Thread, Lock
from flask import stream_with_context, request, Response


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


class Streamer(Thread):
    def __init__(self, stream, max_clients=3):
        super(Streamer, self).__init__()
        self.__lock = Lock()
        self.__stream = stream
        self.__max_clients = max_clients
        self.__clients = [None] * max_clients
        self.__clients_count = 0
        self.__streamed = False

    def add_client(self, client):
        self.__lock.acquire()
        added = False
        for i in range(0, len(self.__clients)):
            if not self.__clients[i]:
                self.__clients[i] = client
                self.__clients_count += 1
                added = True
                break
        self.__lock.release()
        return added

    def shutdown_close(self):
        self.__lock.acquire()
        for i in range(0, len(self.__clients)):
            if not self.__clients[i]:
                continue
            self.__shutdown_close_client(i)
        self.__lock.release()

    def __shutdown_close_client(self, id):
        print('shutting down client: %d' % id)
        client = self.__clients[id]
        print('shutting down client: %s' % str(client))
        sock = client[0]
        try:
            sock.shutdown(socket.SHUT_RDWR)
            print('client shotdown: %d' % id)
        except Exception as e:
            print(e)
        try:
            sock.close()
            print('client closed: %d' % id)
        except Exception as e:
            print(e)
        self.__clients[id] = None
        self.__clients_count -= 1

    def run(self):
        while True:
            data = self.__stream.recv(4096)
            self.__lock.acquire()
            if self.__streamed and self.__clients_count <= 0:
                print('all clients disconnected. quitting ...')
                self.__lock.release()
                return
            for i in range(0, len(self.__clients)):
                if not self.__clients[i]:
                    continue
                sock = self.__clients[i][0]
                try:
                    self.__streamed = True
                    sock.send(data)
                except Exception as e:
                    print(e)
                    self.__shutdown_close_client(i)
            self.__lock.release()


# accepting connections
def pipe(host_in, port_in, host_out, port_out):
    # INPUT
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_in.connect((host_in, port_in))
    streamer = Streamer(sock_in)
    streamer.start()
    # OUTPUT
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_out.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_out.bind((host_out, port_out))
    sock_out.listen(8)
    print('streaming at %s:%s' % (host_out, port_out))
    try:
        while streamer.is_alive():
            client = sock_out.accept()
            if not streamer.add_client(client):
                print('could not accept connection from %s: maximum number of clients reached' % str(client[1]))
                client[0].shutdown(socket.SHUT_RDWR)
                client[0].close()
            else:
                print('accepted connection from %s' % str(client[1]))
    except:
        streamer.shutdown_close()
    print('stopping streaming...')
    streamer.join()
    sock_out.shutdown(socket.SHUT_RDWR)
    sock_out.close()
    sock_in.shutdown(socket.SHUT_RDWR)
    sock_in.close()


def predict_stream(*args):
    """
    Function to make prediction on a stream

    TODO:
    1) Open new output socket and write predictions to it. This method will return just status of creating such socket.
    2) Control the streaming; e.g., stop

    Prerequisities:
        1) DEEPaaS with 'predict_stream' method (similar to 'predict_url')
        2) tunnel the stream locally: ssh -q -f -L 9999:127.0.0.1:9999 deeplogs 'tail -F /storage/bro/logs/current/conn.log | nc -l -k 9999'
        3) call DEEPaaS predict_stream with json string parameter:
            {
                "in": {
                    "host": "127.0.0.1",
                    "port": 9999,
                    "encoding": "utf-8",
                    "columns": [16, 9]
                },
                "out": {
                    "host": "127.0.0.1",
                    "port": 9998,
                    "encoding": "utf-8",
                }
            }
    """
    print('args: %s' % args)
    # return Response(stream_with_context(bullshit(100)))

    params = json.loads(args[0])

    # INPUT params
    params_in = params['in']
    host_in = params_in['host']
    port_in = int(params_in['port'])
    encoding_in = params_in['encoding']

    # OUTPUT params
    params_out = params['out']
    host_out = params_out['host']
    port_out = int(params_out['port'])
    encoding_out = params_out['encoding']

    return pipe(host_in, port_in, host_out, port_out)

    # INPUT socket
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print('connecting to %s:%s' % (host_in, port_in))
        sock_in.connect((host_in, port_in))
        print('successfully connected')
    except Exception as e:
        message = str(e)
        return message

    # OUTPUT socket
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print('connecting to %s:%s' % (host_out, port_out))
        sock_out.connect((host_out, port_out))
        print('successfully connected')
    except Exception as e:
        message = str(e)
        return message

    chunks = []
    chunks_to_join = 20
    chunks_collected = 0
    chunk_size = 512
    receiving = True
    predictions_total = 0

    seq_len = mods_model.get_sequence_len()
    buffer = pd.DataFrame()

    while receiving:
        chunk = sock_in.recv(chunk_size)
        print('chunk: %d B' % len(chunk))
        if not chunk:
            # end of the stream
            print('end of the input stream')
            receiving = False
        else:
            chunks.append(chunk)
            chunks_collected += 1
            print('chunks: %d' % chunks_collected)
        if chunks_collected == chunks_to_join or not receiving:
            # join collected chunks
            raw = b''.join(chunks)
            # the beginning of the complete data
            beg = raw.find(b'\n') + 1
            # the end of the complete data
            end = raw.rfind(b'\n') + 1
            # completed raw lines
            raw_complete = raw[beg:end]
            # store the incomplete line for the next loop
            chunks = [raw[end:]]
            chunks_collected = 0
            # create pandas dataframe
            df = pd.read_csv(
                io.BytesIO(raw_complete),
                sep='\t',
                header=None,
                usecols=params_in['columns'],
                skiprows=0,
                skipfooter=0,
                engine='python',
                skip_blank_lines=True
            )
            print('new rows: %d rows' % len(df))
            for col in df:
                df[col] = [np.nan if x == '-' else pd.to_numeric(x, errors='coerce') for x in df[col]]
            # append the dataframe to the buffer
            buffer = buffer.append(df)
            print('buffer: %d rows' % len(buffer))
        # there must be at least 'seq_len' rows in the buffer to make time series generator and predict
        if len(buffer) >= seq_len:
            predictions = mods_model.predict(buffer)
            predictions_total += 1
            buffer = pd.DataFrame()
            message = json.dumps({'status': 'ok', 'predictions': predictions.tolist()})
            print(message)
            tsv = pd.DataFrame(predictions).to_csv(None, sep="\t", index=False, header=False)
            tsv = tsv.encode(encoding_out)
            try:
                sock_out.send(tsv)
            except Exception as e:
                print(e)
                receiving = False
    sock_out.shutdown()
    sock_out.close()
    sock_in.shutdown()
    sock_in.close()
    return {
        'status': 'ok',
        'predictions_total': predictions_total
    }


def train(*args):
    """
    Train network
    """
    message = 'Not implemented in the model (train)'
    return message
