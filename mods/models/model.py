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

import argparse
import io
import json
import os
import socket
import sys
import time

import numpy as np
import pandas as pd
import pkg_resources

# import project config.py
import mods.config as cfg
import mods.dataset.make_dataset as mdata
import mods.models.mods_model as MODS

# import utilities

# load model
# model_filename = os.path.join(cfg.app_models, cfg.default_model)
# mods_model = MODS.mods_model(model_filename)
#
# if not mods_model:
#     print('Could not load model: %s' % model_filename)
#     sys.exit(1)

mods_model = None


def get_model(model_filename=os.path.join(cfg.app_models, cfg.default_model)):
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
            predictions = get_model().predict(df)
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
    print('---\nargs:\n%s\n---' % args)

    args = args[0]

    # uncomment to get data via rclone
    # mdata.prepare_data()

    m = MODS.mods_model(os.path.join(cfg.app_models, args.model_name))

    # TODO: read path from user input
    df_train_path = os.path.join(cfg.app_data, 'features-20180414-20181015-win-1_hour-slide-10_minutes.tsv')

    # TODO: add 'usecols' in args and parse it
    df_train = m.load_data(
        path=df_train_path,
        usecols=['number_of_conn', 'sum_orig_kbytes']
    )
    print(df_train)

    m.train(
        df_train=df_train,
        df_test=None,
        epochs=args.epochs,
        sequence_len=args.sequence_len,
        multivariate=args.multivariate,
        model_type=args.model_type
    )

    m.save(os.path.join(cfg.app_models, args.model_name))

    return message


def get_train_args():
    return {
        'model_name': {
            'default': 'model.zip',
            'help': 'e.g. model.zip',
            'required': True
        },
        'multivariate': {
            'default': cfg.multivariate,
            'help': '',
            'required': True
        },
        'sequence_len': {
            'default': cfg.sequence_len,
            'help': '',
            'required': True
        },
        # 'model_delta': {
        #     'default': cfg.model_delta,
        #     'help': '',
        #     'required': True
        # },
        # 'interpolate': {
        #     'default': cfg.interpolate,
        #     'help': '',
        #     'required': True
        # },
        'model_type': {
            'default': cfg.model_type,
            'help': '',
            'required': True
        },
        'epochs': {
            'default': cfg.epochs,
            'help': '',
            'required': True
        },
        'epochs_patience': {
            'default': cfg.epochs_patience,
            'help': '',
            'required': True
        },
        'blocks': {
            'default': cfg.blocks,
            'help': '',
            'required': True
        }
    }


# during development it might be practical
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """

    if args.method == 'get_metadata':
        get_metadata()
    elif args.method == 'predict_file':
        predict_file(args.file)
    elif args.method == 'predict_data':
        predict_data(args.file)
    elif args.method == 'predict_url':
        predict_url(args.url)
    elif args.method == 'train':
        start = time.time()
        train(args.n_epochs)
        print("Elapsed time:  ", time.time() - start)
    else:
        get_metadata()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    parser.add_argument('--file', type=str, help='File to do prediction on, full path')
    parser.add_argument('--url', type=str, help='URL with the image to do prediction on')
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Number of epochs to train on')
    args = parser.parse_args()

    main()
