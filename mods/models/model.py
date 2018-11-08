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

# cat data/features/serie_mock_1h_10m.tsv | cut -f 4-5 -d $'\t' | tail -n +2 | head -n+1 -n 10 | awk '{printf "[%s, %s],\n", $1, $2}'
sample_data = np.array([
    [16595, 484293804],
    [29823, 2352139701],
    [42214, 3441880182],
    [55848, 3597720209],
    [103089, 3608747597],
    [116832, 3612003592],
    [114082, 3130504652],
    [115091, 3862441600],
    [115942, 3159371336],
    [117386, 3005934068],
])

model_filename = os.path.join(cfg.app_models, cfg.default_model)
model_metadata_filename = utl.get_metadata_filename(model_filename)

# load model
model, model_metadata = utl.load_model(model_filename, model_metadata_filename)
if not model:
    print('Could not load model: %s' % model_filename)
    sys.exit(1)
elif not model_metadata:
    print('Could not load model metadata: %s' % model_metadata_filename)
    sys.exit(1)
else:
    print('Model loaded: %s' % model_filename)

model_init()


# todo
def get_sample_data():
    global sample_data
    return [json.dumps({
        'labels': sample_data[:, 0].tolist(),
        'rows': sample_data[:, 1:].tolist()
    })]


# todo: model initialization
def model_init():
    # x = K.zeros((metadata['sequence_len'], metadata['multivariate']), dtype='float32')
    # model.predict(np.zeros((1, model_metadata['multivariate'])))
    model.predict(x)


def transform(data):
    # First order differential for numpy array      y' = d(y)/d(t) = f(y,t)
    # be carefull                                   len(dt) == len(data)-1
    dt = utl.delta_timeseries(data)
    return dt


# returns time series generator
def get_tsg(data, length):
    return TimeseriesGenerator(data, data, length=sequence_len, sampling_rate=1, stride=1, batch_size=1)


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
