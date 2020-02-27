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
Created on Mon Nov 07 07:44:07 2019

Train models with first order differential to monitor changes

@author: stefan dlugolinsky
@author: giang nguyen
"""

import os

import logging
import pkg_resources
from distutils.file_util import copy_file
from keras import backend
from marshmallow import Schema, INCLUDE
from webargs import fields

import mods.config as cfg
import mods.dataset.make_dataset as mdata
import mods.models.mods_model as MODS
import mods.utils as utl
from mods.mods_types import TimeRange


class TimeRangeField(fields.Field):
    def _serialize(self, value: TimeRange, attr, obj, **kwargs):
        return str(value)

    def _deserialize(self, value: str, attr, data, **kwargs):
        return TimeRange.from_str(value)


# @dataclass
# class TrainArgs:
#     model_name: str
#     data_select_query: str
#     train_time_range: TimeRange
#     train_time_ranges_excluded: list
#     test_time_range: TimeRange
#     test_time_ranges_excluded: list
#     window_slide: str
#     sequence_len: int
#     model_delta: bool
#     model_type: str
#     num_epochs: int
#     epochs_patience: int
#     blocks: int
#     steps_ahead: int
#     batch_size: int
#
#
# @dataclass
# class PredictArgs:
#     model_name: str
#     data_select_query: str
#     time_range: TimeRange
#     time_ranges_excluded: list
#     window_slide: str
#     batch_size: int


class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    model_name = fields.Str(
        required=False,
        missing=cfg.model_name,
        description="Name of the model. The model will be saved as a zip file; e.g., <model_name>.zip"
    )
    data_select_query = fields.Str(
        required=False,
        missing=cfg.train_data_select_query,
        description= \
            """
            Query for data selection from the datapool.
            
            format: p1;p2|p2c1|col~p2c2_renamed|...;...#c5,c6,...
                p1, p2, ...     - protocols; e.g., conn, http, ssh
                p2c1, p2c2, ... - columns of the protocol
                ~p2c2_renamed   - rename column
                #c5, c6, ...    - columns to merge data over
            """
    )
    train_time_range = TimeRangeField(
        required=False,
        missing=cfg.train_time_range,
        description= \
            """
            Specify the time range for training; e.g., <2018-01-01,2019-01-01)
            format: LBRACKET YYYY-MM-DD,YYYY-MM-DD RBRACKET
                LBRACKET: left side bracket, closed: < or open: (
                RBRACKET: right side bracket closed: > or open: )
            """
    )
    train_time_ranges_excluded = fields.List(
        TimeRangeField,
        required=False,
        missing=cfg.train_time_range_excluded,
        description= \
            """
            A list of time ranges to skip. See the format for train or test time range.
            If executing training from the command line, use ';' as time range delimiter.
            """
    )
    test_time_range = TimeRangeField(
        required=False,
        missing=cfg.test_time_range,
        description= \
            """
            Specify the time range for testing; e.g., <2019-01-01,2020-01-01)
            format: LBRACKET YYYY-MM-DD,YYYY-MM-DD RBRACKET
                LBRACKET: left side bracket, closed: < or open: (
                RBRACKET: right side bracket closed: > or open: )
            """
    )
    test_time_ranges_excluded = fields.List(
        TimeRangeField,
        required=False,
        missing=cfg.test_time_range_excluded,
        description= \
            """
            A list of time ranges to skip. See the format for train or test time range.
            If executing training from the command line, use ';' as time range delimiter.
            """
    )
    window_slide = fields.Str(
        required=False,
        missing=cfg.train_ws,
        enum=cfg.train_ws_choices,
        description="Window length and slide duration"
    )
    sequence_len = fields.Integer(
        required=False,
        missing=cfg.sequence_len,
        description="length of the training sequence; e.g., 3, 4, 5, 6, 9, 12"
    )
    model_delta = fields.Boolean(
        required=False,
        missing=cfg.model_delta,
        enum=[True, False],
        description="Differential approach to model training"
    )
    model_type = fields.Str(
        required=False,
        missing=cfg.model_type,
        enum=cfg.model_types,
        description="Choose a type of the model"
    )
    num_epochs = fields.Integer(
        required=False,
        missing=cfg.num_epochs,
        description="Number of training epochs"
    )
    epochs_patience = fields.Integer(
        required=False,
        missing=cfg.epochs_patience,
        description=""
    )
    blocks = fields.Integer(
        required=False,
        missing=cfg.blocks,
        description="Dimensionality of the model's output space"
    )
    steps_ahead = fields.Integer(
        required=False,
        missing=cfg.steps_ahead,
        description="Number of steps ahead to predict"
    )
    batch_size = fields.Integer(
        required=False,
        missing=cfg.batch_size,
        description="Training batch size"
    )


class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    model_name = fields.Str(
        required=False,
        missing=cfg.model_name,
        #enum=cfg.list_models(),
        description="Choose model for prediction. See available models in metadata."
    )
    time_range = TimeRangeField(
        required=False,
        missing=cfg.test_time_range,
        description= \
            """
            Specify the time range for training; e.g., <2018-01-01,2019-01-01)
            format: LBRACKET YYYY-MM-DD,YYYY-MM-DD RBRACKET
                LBRACKET: left side bracket, closed: < or open: (
                RBRACKET: right side bracket closed: > or open: )
            """
    )
    time_ranges_excluded = fields.List(
        TimeRangeField,
        required=False,
        missing=cfg.test_time_range_excluded,
        description= \
            """
            A list of time ranges to skip. See the format for prediction time range.
            If executing prediction from the command line, use ';' as time range delimiter.
            """
    )
    batch_size = fields.Integer(
        required=False,
        missing=cfg.batch_size,
        description="Batch size"
    )


def load_model(
        model_name=cfg.model_name,
        models_dir=cfg.app_models
):
    """
    Function loads existing MODS model

    :param model_name: file name of the model ('zip' extension is optional)
    :param models_dir:
    :return: mods.models.mods_model
    """
    backend.clear_session()
    m = MODS.mods_model(model_name)
    m.load(os.path.join(models_dir, model_name))
    return m


def get_metadata():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_metadata
    :return:
    """
    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        "author": "",
        "author-email": "giang.nguyen@savba.sk, stefan.dlugolinsky@savba.sk",
        "description": "",
        "license": "",
        "url": "https://github.com/deephdc/mods",
        "version": "",
        "models": cfg.list_models()
    }

    # override above values by values from PKG-INFO file (loads metadata from setup.cfg)
    for l in pkg.get_metadata_lines("PKG-INFO"):
        llo = l.lower()
        for par in meta:
            if llo.startswith(par.lower() + ":"):
                _, v = l.split(": ", 1)
                meta[par] = v

    return meta


def warm():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.warm
    :return:
    """
    # prepare the data
    if not (os.path.exists(cfg.app_data_features) and os.path.isdir(cfg.app_data_features)):
        mdata.prepare_data()


def get_train_args(**kwargs):
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    https://marshmallow.readthedocs.io/en/latest/api_reference.html#module-marshmallow.fields
    :param kwargs:
    :return:
    """
    return TrainArgsSchema().fields


def train(**kwargs):
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    """
    logging.info("train(**kwargs) - kwargs: %s" % (kwargs))

    # use this schema
    schema = TrainArgsSchema()
    # deserialize key-word arguments
    train_args = schema.load(kwargs)

    logging.info('train_args: %s', train_args)

    models_dir = cfg.app_models
    model_name = train_args['model_name']

    # support full paths for command line calls
    full_paths = train_args['full_paths'] if 'full_paths' in train_args else False
    if full_paths:
        logging.info('full_paths:', full_paths)
        models_dir = os.path.dirname(model_name)
        model_name = os.path.basename(model_name)

    # read train data from the datapool
    df_train, cached_file_train = utl.datapool_read(
        train_args['data_select_query'],
        train_args['train_time_range'],
        train_args['window_slide'],
        train_args['train_time_ranges_excluded'],
        cfg.app_data_features
    )
    # repair the data
    df_train = utl.fix_missing_num_values(df_train)

    # read test data from the datapool
    df_test, cached_file_test = utl.datapool_read(
        train_args['data_select_query'],
        train_args['test_time_range'],
        train_args['window_slide'],
        train_args['test_time_ranges_excluded'],
        cfg.app_data_features
    )
    # repair the data
    df_test = utl.fix_missing_num_values(df_test)

    backend.clear_session()
    model = MODS.mods_model(model_name)
    model.train(
        df_train=df_train,
        sequence_len=train_args['sequence_len'],
        model_delta=train_args['model_delta'],
        model_type=train_args['model_type'],
        num_epochs=train_args['num_epochs'],
        epochs_patience=train_args['epochs_patience'],
        blocks=train_args['blocks'],
        steps_ahead=train_args['steps_ahead'],
        batch_size=train_args['batch_size']
    )

    # evaluate the model
    predictions = model.predict(df_test)
    metrics = utl.compute_metrics(
        df_test[model.get_sequence_len():-train_args['steps_ahead']],
        predictions[:-train_args['steps_ahead']],  # here, we predict # steps_ahead
        model,
    )

    # put computed metrics into the model to be saved in model's zip
    model.update_metrics(metrics)
    # store data select query into the model
    model.set_data_select_query(train_args['data_select_query'])
    # store time ranges
    model.set_train_time_range(train_args['train_time_range'])
    model.set_test_time_range(train_args['test_time_range'])
    # store window_slide into the model
    model.set_window_slide(train_args['window_slide'])
    # store exclusion filters
    model.set_train_time_ranges_excluded(train_args['train_time_ranges_excluded'])
    model.set_test_time_ranges_excluded(train_args['test_time_ranges_excluded'])
 
    # save model locally
    file = model.save(os.path.join(models_dir, model_name))
    
    # copy model to a remote dir
    if cfg.app_models_remote != None:
        logging.info('copy_file(%s, %s): start' % (file, cfg.app_models_remote))
        copy_file(file, cfg.app_models_remote)
        logging.info('copy_file(%s, %s): done' % (file, cfg.app_models_remote))
    else:
        logging.info('skipping uploading model into a remote storage: cfg.app_models_remote=%s' % cfg.app_models_remote)
    
    message = {
        'dir_models': models_dir,
        'model_name': model_name,
        'steps_ahead': model.get_steps_ahead(),
        'batch_size': model.get_batch_size(),
        'window_slide': train_args['window_slide'],
        'data_select_query': train_args['data_select_query'],
        'train_time_range': str(train_args['train_time_range']),
        'train_time_ranges_excluded': str(train_args['train_time_ranges_excluded']),
        'train_cached_df': cached_file_train,
        'test_time_range': str(train_args['test_time_range']),
        'test_time_ranges_excluded': str(train_args['test_time_ranges_excluded']),
        'test_cached_df': cached_file_test,
        'evaluation': model.get_metrics(),
    }

    return message


def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return PredictArgsSchema().fields


def predict(**kwargs):
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.predict
    :param kwargs:
    :return:
    """
    logging.info("predict(**kwargs) - kwargs: %s" % (kwargs))

    # use this schema
    schema = PredictArgsSchema()
    # deserialize key-word arguments
    predict_args = schema.load(kwargs)

    logging.info('predict_args: %s', predict_args)

    models_dir = cfg.app_models
    model_name = predict_args['model_name']

    # support full paths for command line calls
    full_paths = predict_args['full_paths'] if 'full_paths' in predict_args else False
    if full_paths:
        logging.info('full_paths:', full_paths)
        models_dir = os.path.dirname(model_name)
        model_name = os.path.basename(model_name)

    backend.clear_session()
    model = load_model(
        models_dir=models_dir,
        model_name=model_name
    )

    data_select_query = model.get_data_select_query()
    window_slide = model.get_window_slide()

    # read data from the datapool
    df_data, cached_file_train = utl.datapool_read(
        data_select_query,
        predict_args['time_range'],
        window_slide,
        predict_args['time_ranges_excluded'],
        cfg.app_data_features
    )
    # repair the data
    df_data = utl.fix_missing_num_values(df_data)

    # override batch_size
    batch_size = kwargs['batch_size']
    model.set_batch_size(batch_size)

    predictions = model.predict(df_data)

    message = {
        'dir_models': models_dir,
        'model_name': model_name,
        'data_select_query': data_select_query,
        'window_slide': window_slide,
        'time_range': str(predict_args['time_range']),
        'time_ranges_excluded': str(predict_args['time_ranges_excluded']),
        'cached_df': cached_file_train,
        'steps_ahead': model.get_steps_ahead(),
        'batch_size': model.get_batch_size(),
        'evaluation': utl.compute_metrics(
            df_data[model.get_sequence_len():-model.get_steps_ahead()],
            predictions[:-model.get_steps_ahead()],
            model,
        ),
        'predictions': predictions.tolist()
    }

    return message
