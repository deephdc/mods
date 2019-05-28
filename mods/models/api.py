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

    print('predict_file - args: %s' % args)
    print('predict_file - kwargs: %s' % kwargs)

    data_prepared = False

    message = 'Error reading input data'

    if args:
        for arg in args:
            message = {'status': 'ok', 'predictions': []}

            # prepare data
            if not data_prepared:
                bootstrap_data = yaml.safe_load(arg.bootstrap_data)
                if bootstrap_data:
                    mdata.prepare_data()
                    data_prepared = True

            model_name = yaml.safe_load(arg.model_name)
            data_file = yaml.safe_load(arg.file)

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
                if data_file == cfg.data_predict:
                    data_file = os.path.join(cfg.app_data_predict, data_file)
            else:
                data_file = os.path.join(cfg.app_data_predict, data_file)

            m = get_model(
                models_dir=models_dir,
                model_name=model_name
            )

            # override batch_size
            batch_size = yaml.safe_load(arg.batch_size)
            m.set_batch_size(batch_size)

            df_data = m.read_file_or_buffer(
                data_file,
                usecols=usecols,
                sep='\t',
                skiprows=skiprows,
                skipfooter=skipfooter,
                engine='python',
                header=header
            )

            predictions = m.predict(df_data)

            message = {
                'status': 'ok',
                'dir_models': models_dir,
                'model_name': model_name,
                'data': data_file,
                'steps_ahead': m.get_steps_ahead(),
                'batch_size': m.get_batch_size(),
                'usecols': usecols,
                'evaluation': utl.compute_metrics(
                    df_data[m.get_sequence_len():-m.get_steps_ahead()],
                    predictions[:-m.get_steps_ahead()],
                    m,
                )
            }

            message['predictions'] = predictions.tolist()

    return message


def predict_data(*args, **kwargs):
    """
    Function to make prediction on an uploaded file
    """

    print('predict_data - args: %s' % str(args))
    print('predict_data - kwargs: %s' % str(kwargs))

    data_prepared = False

    message = 'Error reading input data'

    if args:
        for arg in args:
            message = {'status': 'ok', 'predictions': []}

            # prepare data
            if not data_prepared:
                bootstrap_data = yaml.safe_load(arg.bootstrap_data)
                if bootstrap_data:
                    mdata.prepare_data()
                    data_prepared = True

            try:
                model_name = yaml.safe_load(arg.model_name)
            except Exception:
                model_name = cfg.model_name

            file_storage = arg.files
            buffer = io.BytesIO(file_storage.read())

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

            m = get_model(
                models_dir=models_dir,
                model_name=model_name
            )

            # override batch_size
            batch_size = yaml.safe_load(arg.batch_size)
            m.set_batch_size(batch_size)

            df_data = m.read_file_or_buffer(
                buffer,
                sep=sep,
                skiprows=skiprows,
                skipfooter=skipfooter,
                engine='python',
                header=header
            )

            predictions = m.predict(df_data)

            message = {
                'status': 'ok',
                'dir_models': models_dir,
                'model_name': model_name,
                'data': file_storage.filename,
                'steps_ahead': m.get_steps_ahead(),
                'batch_size': m.get_batch_size(),
                'evaluation': utl.compute_metrics(
                    df_data[m.get_sequence_len():-m.get_steps_ahead()],
                    predictions[:-m.get_steps_ahead()],
                    m,
                )
            }

            message['predictions'] = predictions.tolist()

    return message


def predict_url(*args, **kwargs):
    """
    Function to make prediction on a local file
    """
    print('predict_url - args: %s' % args)
    print('predict_url - kwargs: %s' % kwargs)
    message = 'Not implemented in the model (predict_url)'
    return message


def predict_stream(*args, **kwargs):
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
    print('predict_stream - args: %s' % args)
    print('predict_stream - kwargs: %s' % kwargs)

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


# !!! deepaas calls get_test_args() to get args for 'predict'
def get_test_args():
    predict_args = cfg.set_predict_args()

    # convert default values and possible 'choices' into strings
    for key, val in predict_args.items():
        val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]
        print(val['default'], type(val['default']))

    return predict_args


def train(args, **kwargs):
    """
    Train network
    """
    print("train(args, **kwargs)\n\targs:\n%s\n\tkwargs:\n%s" % (
        args,
        kwargs
    ))

    # prepare the data
    bootstrap_data = yaml.safe_load(args.bootstrap_data)
    if bootstrap_data:
        mdata.prepare_data()

    # selecting protocols, protocol columns and data merging specification
    data_select_query = yaml.safe_load(args.data_select_query)

    # window + slide
    window_slide = yaml.safe_load(args.window_slide)

    # train - time range
    train_time_range = str(yaml.safe_load(args.train_time_range)).split('--', 1)
    assert train_time_range
    assert len(train_time_range) == 2
    train_time_range_beg = utl.parse_datetime(train_time_range[0])
    train_time_range_end = utl.parse_datetime(train_time_range[1])
    train_time_range = (train_time_range_beg, train_time_range_end)

    # train - time range exclusion filter
    train_time_range_excluded = utl.parse_datetime_ranges(yaml.safe_load(args.train_time_ranges_excluded))

    # test - time range
    test_time_range = str(yaml.safe_load(args.test_time_range)).split('--', 1)
    assert test_time_range
    assert len(test_time_range) == 2
    test_time_range_beg = utl.parse_datetime(test_time_range[0])
    test_time_range_end = utl.parse_datetime(test_time_range[1])
    test_time_range = (test_time_range_beg, test_time_range_end)

    # test - time range exclusion filter
    test_time_range_excluded = utl.parse_datetime_ranges(yaml.safe_load(args.test_time_ranges_excluded))

    model_name = yaml.safe_load(args.model_name)
    sequence_len = yaml.safe_load(args.sequence_len)
    model_delta = yaml.safe_load(args.model_delta)
    interpolate = yaml.safe_load(args.interpolate)
    model_type = yaml.safe_load(args.model_type)
    num_epochs = yaml.safe_load(args.num_epochs)
    epochs_patience = yaml.safe_load(args.epochs_patience)
    blocks = yaml.safe_load(args.blocks)
    steps_ahead = yaml.safe_load(args.steps_ahead)
    batch_size = yaml.safe_load(args.batch_size)

    # support full paths for command line calls
    models_dir = cfg.app_models
    full_paths = kwargs['full_paths'] if 'full_paths' in kwargs else False
    if full_paths:
        models_dir = os.path.dirname(model_name)
        model_name = os.path.basename(model_name)

    # read train data from the datapool
    df_train, cached_file_train = utl.datapool_read(data_select_query, train_time_range, window_slide, train_time_range_excluded, cfg.app_data_features)

    # read test data from the datapool
    df_test, cached_file_test = utl.datapool_read(data_select_query, test_time_range, window_slide, test_time_range_excluded, cfg.app_data_features)

    backend.clear_session()
    model = MODS.mods_model(model_name)
    model.train(
        df_train=df_train,
        multivariate=len(df_train.columns),
        sequence_len=sequence_len,
        model_delta=model_delta,
        interpolate=interpolate,
        model_type=model_type,
        num_epochs=num_epochs,
        epochs_patience=epochs_patience,
        blocks=blocks,
        steps_ahead=steps_ahead,
        batch_size=batch_size
    )

    # evaluate the model
    predictions = model.predict(df_test)
    metrics = utl.compute_metrics(
        df_test[model.get_sequence_len():-steps_ahead],
        predictions[:-steps_ahead],  # here, we predict # steps_ahead
        model,
    )

    # put computed metrics into the model to be saved in model's zip
    model.update_metrics(metrics)

    # save model locally
    file = model.save(os.path.join(models_dir, model_name))
    dir_remote = cfg.app_models_remote

    # upload model using rclone
    out, err = dutils.rclone_call(
        src_path=file,
        dest_dir=dir_remote,
        cmd='copy'
    )
    print('rclone_copy(%s, %s):\nout: %s\nerr: %s' % (file, dir_remote, out, err))

    message = {
        'status': 'ok',
        'dir_models': models_dir,
        'model_name': model_name,
        'steps_ahead': model.get_steps_ahead(),
        'batch_size': model.get_batch_size(),
        'training_time': model.get_training_time(),
        'data_select_query': data_select_query,
        'train_time_range': utl.datetime2str(train_time_range),
        'train_time_range_excluded': utl.datetime2str(train_time_range_excluded),
        'train_cached_df': cached_file_train,
        'test_time_range': utl.datetime2str(test_time_range),
        'test_time_range_excluded': utl.datetime2str(test_time_range_excluded),
        'test_cached_df': cached_file_test,
        'evaluation': model.get_metrics(),
    }

    return message


def test_file(*args, **kwargs):
    """
    Function to make test on a local file
    """

    print("args:", args)
    print("kwargs:", kwargs)

    message = {
        'status': 'error',
        'message': 'Error reading input data'
    }

    if not args:
        return message

    messages = []

    data_prepared = False

    for arg in args:

        if not data_prepared:
            # prepare data
            bootstrap_data = yaml.safe_load(arg.bootstrap_data)
            if bootstrap_data:
                mdata.prepare_data()
                data_prepared = True

        dir_models = cfg.app_models
        model_name = yaml.safe_load(arg.model_name)
        dir_data_test = cfg.app_data_test
        data_test = yaml.safe_load(arg.file)
        usecols = [utl.parse_int_or_str(col) for col in yaml.safe_load(arg.pd_usecols).split(',')]
        skiprows = yaml.safe_load(arg.pd_skiprows)
        skipfooter = yaml.safe_load(arg.pd_skipfooter)
        header = yaml.safe_load(arg.pd_header)

        # support full paths for command line calls
        if 'full_paths' in kwargs and kwargs['full_paths']:
            if model_name != cfg.model_name:
                dir_models = os.path.dirname(model_name)
                model_name = os.path.basename(model_name)
            if data_test != cfg.test_data:
                dir_data_test = os.path.dirname(data_test)
                data_test = os.path.basename(data_test)

        # important!
        backend.clear_session()

        # load model
        file_model = os.path.join(
            dir_models,
            model_name
        ) + ('.zip' if not model_name.lower().endswith('.zip') else '')
        m = MODS.mods_model(model_name)
        m.load(file_model)

        # override batch_size
        batch_size = yaml.safe_load(arg.batch_size)
        m.set_batch_size(batch_size)

        # load data
        df_test = m.load_data(
            path=os.path.join(dir_data_test, data_test),
            usecols=usecols,
            skiprows=skiprows,
            skipfooter=skipfooter,
            header=header
        )

        # predict
        predictions = m.predict(df_test)

        messages.append({
            'status': 'ok',
            'dir_models': dir_models,
            'model_name': model_name,
            'test_data': data_test,
            'steps_ahead': m.get_steps_ahead(),
            'batch_size': m.get_batch_size(),
            'usecols': usecols,
            'evaluation': utl.compute_metrics(
                df_test[m.get_sequence_len():-m.get_steps_ahead()],
                predictions[:-m.get_steps_ahead()],
                m,
            )
        })

    return messages
