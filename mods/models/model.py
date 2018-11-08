# -*- coding: utf-8 -*-
"""
Model description
"""
import pkg_resources
# import project config.py
import mods.config as cfg
# import utilities
import mods.utils as utl

import os
import sys
import json
import numpy as np
import pandas as pd
import keras

from zipfile import ZipFile


class MODSModel:
    def __init__(self, file):
        self.config = None
        self.model = None
        self.scaler = None
        self.sample_data = None
        self.__load(file)
        self.__init()

    def __load(self, file):
        print('Loading model: %s' % file)
        with ZipFile(file) as zip:
            self.__load_config(zip, 'config.json')
            self.__load_model(zip, self.config['model'])
            self.__load_scaler(zip, self.config['scaler'])
            self.__load_sample_data(zip, self.config['sample_data'])
        print('Model loaded')

    def __load_config(self, zip, file):
        print('Loading model config')
        with zip.open(file) as f:
            self.config = json.load(f)
        print('Model config:\n%s' % json.dumps(self.config, indent=True))

    def __load_model(self, zip, config):
        print('Loading keras model')
        with zip.open(config['file']) as f:
            contents = zip.read(f)
            self.model = keras.models.load_model(contents)
        print('Keras model loaded')

    def __load_scaler(self, zip, config):
        print('Loading scaler')
        with zip.open(config['file']) as f:
            self.scaler = joblib.load(f)
        print('Scaler loaded')

    def __load_sample_data(self, zip, config):
        print('Loading sample data')
        with zip.open(config['file']) as f:
            pd.read_csv(f,
                        sep=config['sep'],
                        skiprows=config['skiprows'],
                        skipfooter=config['skipfooter'],
                        engine=config['engine'],
                        usecols=lambda col: col in config['usecols']
                        )
        print('Sample data loaded')

    def __init(self):
        print('Initializing model')
        return


# load model
model_filename = os.path.join(cfg.app_models, cfg.default_model)
mods_model = MODSModel(model_filename)

if not mods_model:
    print('Could not load model: %s' % model_filename)
    sys.exit(1)


def transform(data):
    # First order differential for numpy array      y' = d(y)/d(t) = f(y,t)
    # be carefull                                   len(dt) == len(data)-1
    dt = utl.delta_timeseries(data)
    return dt


def get_sample_data():
    return sample_data()
    # df = pd.DataFrame(sample_data)
    # df.interpolate(inplace=True)
    # df = df.values.astype('float32')
    # df = transform(df)
    # return df


# todo: model initialization
def model_init():
    df = get_sample_data()
    tsg = get_tsg(df, model_metadata['sequence_len'])
    pred = model.predict_generator(tsg)
    data = denormalize(model, scaler, pred)
    print(data)


model_init()


# returns time series generator
def get_tsg(data, length):
    return TimeseriesGenerator(data, data, length=length, sampling_rate=1, stride=1, batch_size=1)


# normalizes data
def normalize(data):
    # Scale all metrics but each separately
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data, scaler


# denormalizes time series
def denormalize(model, scaler, tsg):
    return scaler.inverse_transform(model.predict_generator(tsg))


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

            data_json = json.loads(data[0])
            rows = np.array(data_json['rows'], dtype=np.float32)

            global model
            predictions = model.predict(rows)

            i = 0
            for prediction in predictions:
                p = {
                    'prob': float(prediction[0])
                }
                if 'labels' in data_json:
                    p['label'] = data_json['labels'][i]
                message.get('predictions').append(p)
                i = i + 1
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
