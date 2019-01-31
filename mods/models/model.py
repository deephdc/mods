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

import io
import json
import os
import socket

import numpy as np
import pandas as pd
import pkg_resources

# import project config.py
import mods.config as cfg
import mods.models.mods_model as MODS
import mods.utils as utl

# import utilities

# load model
# model_filename = os.path.join(cfg.app_models, cfg.default_model)
# mods_model = MODS.mods_model(model_filename)
#
# if not mods_model:
#     print('Could not load model: %s' % model_filename)
#     sys.exit(1)

mods_model = None


def get_model(model_filename=os.path.join(cfg.app_models, cfg.model_name)):
    global mods_model
    if not mods_model:
        mods_model = MODS.mods_model(model_filename)
    return mods_model


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
    message = 'Error reading input data'
    if args:
        for file in args:
            message = {'status': 'ok', 'predictions': []}
            predictions = get_model().predict_file_or_buffer(
                file,
                sep='\t',
                skiprows=0,
                skipfooter=0,
                engine='python'
            )
            message['predictions'] = predictions.tolist()
    return message


def predict_data(*args):
    """
    Function to make prediction on an uploaded file
    """
    message = 'Error reading input data'
    if args:
        for data in args:
            message = {'status': 'ok', 'predictions': []}
            buffer = io.BytesIO(data[0])
            predictions = get_model().predict_file_or_buffer(
                buffer,
                sep='\t',
                skiprows=0,
                skipfooter=0,
                engine='python'
            )
            message['predictions'] = predictions.tolist()
    return message


def predict_url(*args):
    """
    Function to make prediction on a local file
    """
    message = 'Not implemented in the model (predict_url)'
    return message


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
                    "encoding": "utf-8"
                }
            }
    """
    print('args: %s' % args)

    params = json.loads(args[0])

    # INPUT
    params_in = params['in']
    host_in = params_in['host']
    port_in = int(params_in['port'])
    encoding_in = params_in['encoding']
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_in.connect((host_in, port_in))

    # OUTPUT
    params_out = params['out']
    host_out = params_out['host']
    port_out = int(params_out['port'])
    encoding_out = params_out['encoding']
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_out.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_out.bind((host_out, port_out))
    sock_out.listen(1)

    print('streaming at %s:%s' % (host_out, port_out))
    client = sock_out.accept()
    print('accepted connection from %s' % str(client[1]))

    raw = b''
    seq_len = get_model().get_sequence_len()
    buffer = pd.DataFrame()
    predictions_total = 0

    while True:

        recvd = sock_in.recv(4096)
        print('recvd: %d B' % len(recvd))

        if not recvd:
            print('no data received from the input stream')
            break

        # join previous incomplete line with currently received data
        print('raw: %d B' % len(raw))
        raw = raw + recvd
        print('raw + recvd: %d B' % len(raw))

        # the beginning of the complete data
        beg = raw.find(b'\n') + 1
        # the end of the complete data
        end = raw.rfind(b'\n') + 1
        print('(beg, end): (%d, %d)' % (beg, end))

        if beg == end:
            # not enough lines
            continue

        # completed raw lines
        raw_complete = raw[beg:end]
        print('raw_complete: %d B' % len(raw_complete))

        # store the incomplete line for the next loop
        raw = raw[end:]
        print('raw (to the next step): %d B' % len(raw))

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

        # there must be more than 'seq_len' rows in the buffer to make time series generator and predict
        # todo: check the '> seq_len + 1', not sure about it. '> seq_len' gives error:
        # ValueError: `start_index+length=6 > end_index=5` is disallowed, as no part of the sequence would be left to be used as current step.
        if len(buffer) > seq_len + 1:

            # PREDICT
            print('predicting on %d rows' % len(buffer))
            predictions = get_model().predict(buffer)
            predictions_total += 1
            buffer = pd.DataFrame()

            # STDOUT message
            message = json.dumps({'status': 'ok', 'predictions': predictions.tolist()})
            print(message)

            # stream output
            tsv = pd.DataFrame(predictions).to_csv(None, sep="\t", index=False, header=False)
            tsv = tsv.encode(encoding_out)

            try:
                client[0].send(tsv)
            except Exception as e:
                print('could not send data to client: %s' % e)
                break

    print('stopping streaming...')
    client[0].close()

    sock_out.shutdown(socket.SHUT_RDWR)
    sock_out.close()

    sock_in.shutdown(socket.SHUT_RDWR)
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
    args = args[0]
    print('---\nargs:\n%s\n---' % args)

    # uncomment to get data via rclone
    # mdata.prepare_data()

    # model name
    if 'model' not in args:
        args['model'] = cfg.model_name
    if not args['model'].endswith('.zip'):
        args['model'] += '.zip'

    # models directory
    if 'dir_models' not in args:
        args['dir_models'] = cfg.app_models

    m = MODS.mods_model(os.path.join(args['dir_models'], args['model']))

    # data directory
    if 'dir_data' not in args:
        args['dir_data'] = cfg.app_data

    # training data location
    if 'data' not in args:
        args['data'] = cfg.data_train

    # column names to use in training data
    if 'usecols' not in args:
        args['usecols'] = cfg.usecols
    if isinstance(args['usecols'], str):
        args['usecols'] = [
            utl.parse_int_or_str(col)
            for col in args['usecols'].split(',')
        ]
    if 'header' not in args:
        args['header'] = cfg.header

    # loading training data
    df_train = m.load_data(
        path=os.path.join(args['dir_data'], args['data']),
        usecols=args['usecols']
    )

    if 'multivariate' not in args:
        args['multivariate'] = cfg.multivariate
    if 'sequence_len' not in args:
        args['sequence_len'] = cfg.sequence_len
    if 'model_delta' not in args:
        args['model_delta'] = cfg.model_delta
    if 'interpolate' not in args:
        args['interpolate'] = cfg.interpolate
    if 'model_type' not in args:
        args['model_type'] = cfg.model_type
    if 'n_epochs' not in args:
        args['n_epochs'] = cfg.n_epochs
    if 'epochs_patience' not in args:
        args['epochs_patience'] = cfg.epochs_patience
    if 'blocks' not in args:
        args['blocks'] = cfg.blocks

    m.train(
        df_train=df_train,
        multivariate=int(args['multivariate']),
        sequence_len=int(args['sequence_len']),
        model_delta=bool(args['model_delta']),
        interpolate=bool(args['interpolate']),
        model_type=str(args['model_type']),
        n_epochs=int(args['n_epochs']),
        epochs_patience=int(args['epochs_patience']),
        blocks=int(args['blocks'])
    )

    m.save()

    return message


def get_train_args():
    args = {
        'data': {
            'default': cfg.data_train,
            'help': cfg.data_train_help,
            'required': True
        },
        'model': {
            'default': cfg.model_name,
            'help': cfg.model_name_help,
            'required': True
        },
        'multivariate': {
            'default': cfg.multivariate,
            'help': cfg.multivariate_help,
            'required': True
        },
        'sequence_len': {
            'default': cfg.sequence_len,
            'help': cfg.sequence_len_help,
            'required': True
        },
        'model_delta': {
            'default': cfg.model_delta,
            'help': cfg.model_delta_help,
            'required': True
        },
        'interpolate': {
            'default': cfg.interpolate,
            'help': cfg.interpolate_help,
            'required': True
        },
        'model_type': {
            'default': cfg.model_type,
            'help': cfg.model_type_help,
            'required': True
        },
        'n_epochs': {
            'default': cfg.n_epochs,
            'help': cfg.n_epochs_help,
            'required': True
        },
        'epochs_patience': {
            'default': cfg.epochs_patience,
            'help': cfg.epochs_patience_help,
            'required': True
        },
        'blocks': {
            'default': cfg.blocks,
            'help': cfg.blocks_help,
            'required': True
        },
        'usecols': {
            'default': cfg.usecols,
            'help': cfg.usecols_help,
            'required': True
        }
    }
    return {}
