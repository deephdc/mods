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

@author: stefan dlugolinsky
@author: giang nguyen
"""

import io
import json
import os
import socket

import numpy as np
import pandas as pd
import pkg_resources
import yaml
from keras import backend

# import project config.py
import mods.config as cfg
import mods.dataset.data_utils as dutils
import mods.dataset.make_dataset as mdata
import mods.models.mods_model as MODS
import mods.utils as utl


def get_model(
        model_name=cfg.model_name,
        models_dir=cfg.app_models
):
    backend.clear_session()
    m = MODS.mods_model(model_name)
    m.load(os.path.join(models_dir, model_name))
    return m


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


def predict_file(*args, **kwargs):
    """
    Function to make prediction on a local file
    """

    mdata.prepare_data()

    message = 'Error reading input data'

    if args:
        for arg in args:
            message = {'status': 'ok', 'predictions': []}
            model_name = yaml.safe_load(arg.model_name)

            data = yaml.safe_load(arg.file)

            usecols = [utl.parse_int_or_str(col) for col in yaml.safe_load(arg.pd_usecols).split(',')]
            skiprows = yaml.safe_load(arg.pd_skiprows)
            skipfooter = yaml.safe_load(arg.pd_skipfooter)
            header = yaml.safe_load(arg.pd_header)

            # support full paths for command line calls
            models_dir = cfg.app_models
            full_paths = kwargs['full_paths'] if 'full_paths' in kwargs else False

            if full_paths:
                if model_name == cfg.model_name:
                    models_dir = cfg.app_models
                else:
                    models_dir = os.path.dirname(model_name)
                    model_name = os.path.basename(model_name)
                if data == cfg.data_predict:
                    data = os.path.join(cfg.app_data_predict, data)
            else:
                data = os.path.join(cfg.app_data_predict, data)

            predictions = get_model(
                models_dir=models_dir,
                model_name=model_name
            ).predict_file_or_buffer(
                data,
                usecols=usecols,
                sep='\t',
                skiprows=skiprows,
                skipfooter=skipfooter,
                engine='python',
                header=header
            )

            message['predictions'] = predictions.tolist()

    return message


def predict_data(*args, **kwargs):
    """
    Function to make prediction on an uploaded file
    """

    mdata.prepare_data()

    message = 'Error reading input data'

    if args:
        for arg in args:
            message = {'status': 'ok', 'predictions': []}

            try:
                model_name = yaml.safe_load(arg.model_name)
            except Exception:
                model_name = cfg.model_name

            buffer = io.BytesIO(arg[0])

            usecols = cfg.pd_usecols
            sep = cfg.pd_sep
            skiprows = cfg.pd_skiprows
            skipfooter = cfg.pd_skipfooter
            header = cfg.pd_header

            # support full paths for command line calls
            models_dir = cfg.app_models
            full_paths = kwargs['full_paths'] if 'full_paths' in kwargs else False
            if full_paths:
                if model_name == cfg.model_name:
                    models_dir = cfg.app_models
                else:
                    models_dir = os.path.dirname(model_name)
                    model_name = os.path.basename(model_name)

            predictions = get_model(
                models_dir=models_dir,
                model_name=model_name
            ).predict_file_or_buffer(
                buffer,
                usecols=usecols,
                sep=sep,
                skiprows=skiprows,
                skipfooter=skipfooter,
                engine='python',
                header=header
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


def get_train_args():
    train_args = cfg.set_train_args()

    # convert default values and possible 'choices' into strings
    for key, val in train_args.items():
        val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]
        print(val['default'], type(val['default']))

    return train_args


def get_predict_args():
    predict_args = cfg.set_predict_args()

    # convert default values and possible 'choices' into strings
    for key, val in predict_args.items():
        val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]
        print(val['default'], type(val['default']))

    return predict_args


def train(train_args, **kwargs):
    """
    Train network
    """

    print("train_args:", train_args)
    print("kwargs:", kwargs)

    data = yaml.safe_load(train_args.data)
    model_name = yaml.safe_load(train_args.model_name)
    multivariate = yaml.safe_load(train_args.multivariate)
    sequence_len = yaml.safe_load(train_args.sequence_len)
    model_delta = yaml.safe_load(train_args.model_delta)
    interpolate = yaml.safe_load(train_args.interpolate)
    model_type = yaml.safe_load(train_args.model_type)
    num_epochs = yaml.safe_load(train_args.num_epochs)
    epochs_patience = yaml.safe_load(train_args.epochs_patience)
    blocks = yaml.safe_load(train_args.blocks)
    pd_usecols = [utl.parse_int_or_str(col) for col in yaml.safe_load(train_args.pd_usecols).split(',')]
    pd_header = yaml.safe_load(train_args.pd_header)

    # support full paths for command line calls
    models_dir = cfg.app_models
    full_paths = kwargs['full_paths'] if 'full_paths' in kwargs else False
    if full_paths:
        models_dir = os.path.dirname(model_name)
        model_name = os.path.basename(model_name)
        if data == cfg.data_train:
            data = os.path.join(cfg.app_data_features, data)
    else:
        data = os.path.join(cfg.app_data_features, data)

    mdata.prepare_data()

    backend.clear_session()
    m = MODS.mods_model(model_name)

    # loading training data
    df_train = m.load_data(
        path=data,
        usecols=pd_usecols,
        header=pd_header
    )

    m.train(
        df_train=df_train,
        multivariate=multivariate,
        sequence_len=sequence_len,
        model_delta=model_delta,
        interpolate=interpolate,
        model_type=model_type,
        num_epochs=num_epochs,
        epochs_patience=epochs_patience,
        blocks=blocks
    )

    # save model locally
    file = m.save(os.path.join(models_dir, model_name))
    dir_remote = cfg.app_models_remote

    # upload model using rclone
    out, err = dutils.rclone_call(
        src_path=file,
        dest_dir=dir_remote,
        cmd='copy'
    )
    print('rclone_copy(%s, %s):\nout: %s\nerr: %s' % (file, dir_remote, out, err))

    return "OK"


def test_file(*args, **kwargs):
    """
    Function to make test on a local file
    """

    # prepare data
    mdata.prepare_data()

    message = {
        'status': 'error',
        'message': 'Error reading input data'
    }

    if not args:
        return message

    messages = []

    for arg in args:

        dir_models = cfg.app_models
        model_name = yaml.safe_load(arg.model_name)
        dir_data = cfg.app_data_test
        data = yaml.safe_load(arg.file)
        usecols = [utl.parse_int_or_str(col) for col in yaml.safe_load(arg.pd_usecols).split(',')]
        skiprows = yaml.safe_load(arg.pd_skiprows)
        skipfooter = yaml.safe_load(arg.pd_skipfooter)
        header = yaml.safe_load(arg.pd_header)

        # support full paths for command line calls
        if 'full_paths' in kwargs and kwargs['full_paths']:
            if model_name != cfg.model_name:
                dir_models = os.path.dirname(model_name)
                model_name = os.path.basename(model_name)
            if data != cfg.data_test:
                dir_data = os.path.dirname(data)
                data = os.path.basename(data)

        # important!
        backend.clear_session()

        # load model
        file_model = os.path.join(
            dir_models,
            model_name
        ) + ('.zip' if not model_name.lower().endswith('.zip') else '')
        m = MODS.mods_model(model_name)
        m.load(file_model)

        # load data
        df_test = m.load_data(
            path=os.path.join(dir_data, data),
            usecols=usecols,
            skiprows=skiprows,
            skipfooter=skipfooter,
            header=header
        )

        eval_result = m.eval(df_test)

        i = 0
        result = {}
        for metric in m.model.metrics_names:
            result[metric] = eval_result[i]
            i += 1

        messages.append({
            'status': 'ok',
            'dir_models': dir_models,
            'model_name': model_name,
            'test_data': data,
            'usecols': usecols,
            'evaluation': result
        })

    return messages
