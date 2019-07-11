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
Created on 20-03-2019

Train multi-variate models

@author: giangnguyen
"""

import mods.config as cfg
import mods.utils as utl

import datetime
import glob
import os
import time
import numpy as np

from os.path import basename

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import CuDNNGRU
from keras.layers import CuDNNLSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.regularizers import l1
from keras.regularizers import l2
from keras.utils import plot_model

from keras_self_attention import SeqSelfAttention
from tcn import TCN

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.style.use('ggplot')
sns.set('paper', 'darkgrid', color_codes=True)
# %matplotlib inline
# matplotlib.pyplot.switch_backend('agg')


# data is numpy array
def transform(data, model_delta=cfg.model_delta):
    if model_delta:
        # First order differential for numpy array      y' = d(y)/d(t) = f(y,t) 
        # be carefull                                   len(dt) == len(data)-1
        trans_data = data[1:] - data[:-1]
    else:
        # bucketing, taxo, fuzzy
        trans_data = data
        
    return trans_data


def normalize(trans_train, trans_test):
    # Scale all metrics but each separately
    scaler = MinMaxScaler(feature_range=(0, 1))

    norm_train = scaler.fit_transform(trans_train)
    norm_test  = scaler.transform(trans_test)

    # Remove peaks
    # if remove_peak:
    #     norm_test = np.clip(norm_test, a_min=0, a_max=1)
    #     trans_test = scaler.inverse_transform(norm_test)

    # utl.plot_series(norm_test,  'norm_test')
    # utl.plot_series(norm_train, 'norm_train')
    print('\nnormalize:', norm_train.shape, norm_test.shape)

    return scaler, norm_train, norm_test, trans_test


# Generate time series data
def make_timeseries(data,
                    sequence_len=cfg.sequence_len,
                    steps_ahead=cfg.steps_ahead,
                    batch_size=cfg.batch_size
                    ):
    if steps_ahead > 1:
        data_x = data[:-(steps_ahead-1)]
        data_y = data[steps_ahead-1:]
    else:
        data_x = data_y = data

    tsg_data = TimeseriesGenerator(data_x,
                                   data_y,
                                   length=sequence_len,
                                   sampling_rate=1,
                                   stride=1,
                                   batch_size=batch_size
                                   )
    x, y = tsg_data[0]
    print('\ttsg x.shape=', x.shape, '\n\tx=', x, '\n\ttsg y.shape=', y.shape, '\n\ty=', y)
    return tsg_data


def compile_fit_save(model, model_name, tsg_train):
    # Optimizer
    opt = Adam(clipnorm=1.0, clipvalue=0.5)

    # Compile model
    model.compile(loss='mean_squared_error',            # Adam
                  optimizer=opt,                        # 'adam', 'adagrad', 'rmsprop', opt
                  metrics=['mse', 'mae'])               # 'cosine', 'mape'

    # Checkpoints
    filepath = cfg.app_checkpoints + basename(model_name) + '-{epoch:02d}.hdf5'
    checkpoints = ModelCheckpoint(filepath,
                                  monitor='loss',
                                  save_best_only=True,
                                  mode=max,
                                  verbose=1
                                  )
    # Early stopping
    earlystops = EarlyStopping(monitor='loss',
                               patience=cfg.epochs_patience,
                               verbose=1
                               )
    callbacks_list = [checkpoints, earlystops]

    # Fit model on train_tsg
    history = model.fit_generator(tsg_train,
                                  epochs=cfg.num_epochs,
                                  callbacks=callbacks_list
                                  )
    # Plot metrics
    plt.plot(history.history['mean_squared_error'])
    plt.savefig(cfg.app_data + 'metrics-history.png', bbox_inches='tight')
    # plt.show()

    # Save model
    model.save(model_name)
    print('\nSave trained model: ', model_name)

    return model


def create_model(model_name,
                 tsg_train,
                 tsg_test,
                 multivariate,
                 model_type=cfg.model_type,
                 sequence_len=cfg.sequence_len,
                 stacked_blocks=cfg.stacked_blocks,
                 batch_normalization=cfg.batch_normalization,
                 dropout_rate=cfg.dropout_rate
                 ):
    print('Model typ: ', model_type)

    # Define model
    x = Input(shape=(sequence_len, multivariate))

    if model_type == 'MLP':                                         # MLP
        h = Dense(units=multivariate, activation='relu')(x)
        h = Flatten()(h)
    elif model_type == 'autoencoderMLP':                            # autoencoder MLP
        nn = [128, 64, 32, 16, 32, 64, 128]
        h = Dense(units=nn[0], activation='relu')(x)
        for n in nn[1:]:
            h = Dense(units=n, activation='relu')(h)
        h = Flatten()(h)
    elif model_type == 'Conv1D':                                    # CNN
        h = Conv1D(filters=64, kernel_size=2, activation='relu')(x)
        h = MaxPooling1D(pool_size=2)(h)
        h = Flatten()(h)
    elif model_type == 'TCN':                                       # https://pypi.org/project/keras-tcn/
        h = TCN(return_sequences=False)(x)
    elif model_type == 'stackedTCN' and stacked_blocks > 1:         # stacked TCN
        h = TCN(return_sequences=True)(x)
        if stacked_blocks > 2:
            for i in range(stacked_blocks - 2):
                h = TCN(return_sequences=True)(h)
        h = TCN(return_sequences=False)(h)

    if len(K.tensorflow_backend._get_available_gpus()) == 0:    # CPU running
        if model_type == 'GRU':                                     # GRU
            h = GRU(cfg.blocks)(x)
        else:                                                       # default LSTM
            h = LSTM(cfg.blocks)(x)
    else:                                                       # GPU running
        print('Running on GPU')
        if model_type == 'GRU':                                     # GRU
            h = CuDNNGRU(cfg.blocks)(x)
        elif model_type == 'LSTM':                                  # LSTM
            h = CuDNNLSTM(cfg.blocks)(x)
        elif model_type == 'bidirectLSTM':                          # bidirectional LSTM
            h = Bidirectional(CuDNNLSTM(cfg.blocks))(x)
        elif model_type == 'attentionLSTM':                         # https://pypi.org/project/keras-self-attention/
            h = Bidirectional(CuDNNLSTM(cfg.blocks, return_sequences=True))(x)
            h = SeqSelfAttention(attention_activation='sigmoid')(h)
            h = Flatten()(h)
        elif model_type == 'seq2seqLSTM':
            if batch_normalization:                                 # https://leimao.github.io/blog/Batch-Normalization/
                h = CuDNNLSTM(cfg.blocks)(x)
                BatchNormalization()(h)
                h = RepeatVector(sequence_len)(h)
                h = CuDNNLSTM(cfg.blocks)(h)
                BatchNormalization()(h)
            elif 0.0 < dropout_rate < 1.0:                          # dropout
                h = CuDNNLSTM(cfg.blocks)(x)
                h = Dropout(dropout_rate)(h)
                h = RepeatVector(sequence_len)(h)
                h = CuDNNLSTM(cfg.blocks)(h)
                h = Dropout(dropout_rate)(h)
            else:                                                   # seq2seq LSTM
                h = CuDNNLSTM(cfg.blocks)(x)
                h = RepeatVector(sequence_len)(h)
                h = CuDNNLSTM(cfg.blocks)(h)
        elif model_type == 'stackedLSTM' and stacked_blocks > 1:    # stacked LSTM
            h = CuDNNLSTM(cfg.blocks, return_sequences=True)(x)
            if stacked_blocks > 2:
                for i in range(stacked_blocks - 2):
                    h = CuDNNLSTM(cfg.blocks, return_sequences=True)(h)
            h = CuDNNLSTM(cfg.blocks)(x)

    y = Dense(units=multivariate, activation='sigmoid')(h)          # 'softmax' for multiclass classification
    # y = TimeDistributed(Dense(units=multivariate, activation='sigmoid'))(h)

    model = Model(inputs=x, outputs=y)
    print(model.summary())
    # plot_model(model, to_file=model_name+'.png')

    model = compile_fit_save(model, model_name, tsg_train)

    # Evaluate model on test_tsg
    print('\nModel evaluation metrics=', model.metrics_names)
    model_eval = model.evaluate_generator(tsg_test)
    print(model_eval)

    return model, model_eval[1:]


# Invertly transform data and predictions + evaluate on real values
def transform_invert(data,
                     trans,
                     denorm,
                     sequence_len=cfg.sequence_len,
                     model_delta=cfg.model_delta
                     ):
    begin = sequence_len
    end = sequence_len + len(denorm)

    if model_delta:
        Y = data[begin+1: end+1]
        prediction = Y - trans[begin: end] + denorm
    else:
        Y = trans[begin:end]
        prediction = denorm

    print('Tranform_invert -> prediction with real values: ', prediction.shape, Y.shape)
    return prediction, Y


def eval_one_line(results, show_train_eval=False):
    line  = results[0]                                                                          # model_type
    line += '\t' + '\t'.join(str("{0:0.5f}".format(x)) for x in results[1])                     # model_eval: MSE, MAE
    for r in results[2:]:
        line += '\t' + r[0] + '\t'                                                              # SMAPE, R2, COSINE
        if show_train_eval:
            line += '\t train \t'
            line += '\t'.join(x if isinstance(x, str) else str("{0:0.5f}".format(x)) for x in r[1]) # train
            line += '\t test \t'
        line += '\t'.join(x if isinstance(x, str) else str("{0:0.5f}".format(x)) for x in r[2])     # test
    print(line)
    return line


def eval_predictions(pred_train, pred_test,
                     Y_train, Y_test,
                     model_type,
                     model_eval,
                     eval_metrics=cfg.eval_metrics
                     ):
    print('\nEvaluation with real values ')
    results = [model_type, model_eval]

    for m in eval_metrics:
        if m == 'SMAPE':
            err_train = utl.smape(Y_train, pred_train)
            err_test  = utl.smape(Y_test,  pred_test)
        elif m == 'R2':
            err_train = utl.r2(Y_train, pred_train)
            err_test  = utl.r2(Y_test,  pred_test)
        elif m == 'COSINE':
            err_train = utl.cosine(Y_train, pred_train)
            err_test  = utl.cosine(Y_test,  pred_test)
        elif m == 'RMSE':
            err_train = utl.rmse(Y_train, pred_train)
            err_test  = utl.rmse(Y_test, pred_test)
        elif m == 'MAPE':
            err_train = utl.mape(Y_train, pred_train)
            err_test  = utl.mape(Y_test, pred_test)

        results.append([m, err_train, err_test])

    return eval_one_line(results)


# Plot data and predictions
def plot_predictions(pred_train, pred_test,
                     Y_train, Y_test,
                     multivariate,
                     fig_x=cfg.fig_size_x,
                     fig_y=cfg.fig_size_y
                     ):
    fig, ax = plt.subplots(multivariate*2,
                           sharex=True,
                           figsize=(fig_x, multivariate*fig_y))

    for i in range(multivariate):
        print('\nVisualize results: multivariate=', i)
        ax[i*2].plot(Y_train[:,i])
        ax[i*2].plot(pred_train[:,i])
        ax[i*2+1].plot(Y_test[:,i])
        ax[i*2+1].plot(pred_test[:,i])
       
    plt.savefig(cfg.app_data + 'results.png', bbox_inches='tight')
    plt.show()
    return
    

def train_and_test(data_train_filename,
                   data_test_filename,
                   model_type=cfg.model_type,
                   sequence_len=cfg.sequence_len,
                   ahead=cfg.steps_ahead
                   ):
    model_name = utl.get_fullpath_model_name(data_train_filename)
    print(model_name)
    
    if not os.path.exists(cfg.app_checkpoints):
        os.makedirs(cfg.app_checkpoints)
    else:
        for fn in glob.glob(cfg.app_checkpoints + basename(model_name) + '*'):
            print('Remove old checkpoints: ' + fn)
            os.remove(fn) 

    data_train = utl.read_data(data_train_filename)
    data_test  = utl.read_data(data_test_filename)

    # Transform data
    trans_train = transform(data_train)
    trans_test  = transform(data_test)

    # Normalize
    scaler, norm_train, norm_test, trans_test = normalize(trans_train, trans_test)

    # Timeseries
    tsg_train = make_timeseries(norm_train, sequence_len, ahead)
    tsg_test  = make_timeseries(norm_test,  sequence_len, ahead, cfg.batch_size_test)

    multivariate = data_train.shape[1]
    model, model_eval = create_model(model_name, tsg_train, tsg_test, multivariate, model_type, sequence_len)

    # Predict
    model_pred_train = model.predict_generator(tsg_train)
    model_pred_test  = model.predict_generator(tsg_test)

    # Denormalize
    denorm_train = scaler.inverse_transform(model_pred_train)
    denorm_test  = scaler.inverse_transform(model_pred_test)

    # Transform back to real values
    pred_train, Y_train = transform_invert(data_train, trans_train, denorm_train, sequence_len)
    pred_test,  Y_test  = transform_invert(data_test,  trans_test,  denorm_test,  sequence_len)

    # Evaluate with real values
    eval_line = eval_predictions(pred_train, pred_test, Y_train, Y_test, model_type, model_eval)

    # Visualize results
    if cfg.plot:
        plot_predictions(pred_train, pred_test, Y_train, Y_test, multivariate)

    return model_name, eval_line


def wrapper(eval_filename):

    utl.create_files_from_tsv(
        cfg.app_data_train,
        cfg.ws_choice,
        cfg.app_data_features,
        cfg.train_time_range_begin,
        cfg.train_time_range_end,
        cfg.train_time_ranges_excluded
    )
    utl.create_files_from_tsv(
        cfg.app_data_test,
        cfg.ws_choice,
        cfg.app_data_features,
        cfg.test_time_range_begin,
        cfg.test_time_range_end,
        cfg.test_time_ranges_excluded
    )

    utl.merge_files_on_cols(cfg.app_data + cfg.data_filename_train, cfg.app_data_train + cfg.ws_choice)
    utl.merge_files_on_cols(cfg.app_data + cfg.data_filename_test,  cfg.app_data_test  + cfg.ws_choice)

    data_filename_train = cfg.app_data + cfg.data_filename_train
    data_filename_test  = cfg.app_data + cfg.data_filename_test

#    model_types = ['MLP', 'Conv1D', 'autoencoderMLP', 'LSTM', 'GRU', 'bidirectLSTM', 'seq2seqLSTM', 'stackedLSTM', 'attentionLSTM', 'TCN', 'stackedTCN']
    sequence_len = cfg.sequence_len
    repeat_num = cfg.repeat_num

    # for sequence_len in range(6, 20):
    # for steps_ahead in range(1, sequence_len):
    # model_type = 'MLP'
    # for i in range(repeat_num):
    for model_type in cfg.model_types:
        start = time.time()
        model_name, eval_line = train_and_test(data_filename_train, data_filename_test, model_type)
        runtime = str("{0:0.2f}".format(time.time() - start))
        with open(eval_filename, 'a+') as fout:
            fout.write(eval_line + '\tTIMER\t' + runtime + '\n')

    return


if __name__ == "__main__":
    # print(cfg.BASE_DIR)

    # timestamp
    timer_start = datetime.datetime.now()

    ts = str(timer_start).replace(' ', '_').replace('-', '').replace(':', '').split('.')[0]
    eval_fn = cfg.app_data_results + ts + '-' + cfg.eval_filename

    wrapper(eval_fn)

    print(eval_fn)

    # runtime report
    print('\nRuntime =', datetime.datetime.now() - timer_start)
