# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Sat Oct 15 10:27:15 2019

@author: Stefan Dlugolinsky
"""
import unittest

import mods.models.api_v2 as mods_model
from mods import config as cfg
from mods import utils as utl
from mods.models.api_v2 import TrainArgsSchema

debug = True


class TestModelMethods(unittest.TestCase):
    def setUp(self):
        self.meta = mods_model.get_metadata()
        self.train_args = TrainArgsSchema().load({
            'data_select_query': 'conn|in_count_uid~conn_in|out_count_uid~conn_out;' + \
                                 'dns|in_distinct_query~dns_in_distinct;' + \
                                 'ssh|in~ssh_in' + \
                                 '#window_start,window_end',
            'window_slide': 'w01h-s10m',
            'train_time_range': '<2019-06-01,2019-06-03)',
            'train_time_ranges_excluded': ['<2019-06-02,2019-06-03)'],
            'test_time_range': '<2019-06-03,2019-06-04)',
            'test_time_ranges_excluded': [],
            'model_name': 'unit_test',
            'sequence_len': '12',
            'model_delta': 'True',
            'model_type': 'LSTM',
            'num_epochs': '1',
            'epochs_patience': '10',
            'blocks': '12',
            'steps_ahead': '1',
            'batch_size': '1'
        })
        self.app_data_features = cfg.BASE_DIR + '/mods/tests/inputs/datapool/'

    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns list
        """
        self.assertTrue(type(self.meta) is dict)

    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns
        right values (subset)
        """
        self.assertEqual(self.meta['author'], 'Giang Nguyen, Stefan Dlugolinsky')
        self.assertEqual(self.meta['author-email'], 'giang.nguyen@savba.sk, stefan.dlugolinsky@savba.sk')

    def test_datapool_read(self):
        df_train, cached_file_train = utl.datapool_read(
            self.train_args['data_select_query'],
            self.train_args['train_time_range'],
            self.train_args['window_slide'],
            excluded=self.train_args['train_time_ranges_excluded'],
            base_dir=self.app_data_features,
            caching=False
        )
        self.assertEqual(len(df_train), 144)

    def test_api_train(self):
        cfg.app_models_remote = None  # disable remote storage
        cfg.data_pool_caching = False  # disable caching
        cfg.app_data_features = self.app_data_features  # change features dir to test
        msg = mods_model.train(**self.train_args)  # call api to train a model
        print(msg)
        self.assertGreater(msg['evaluation']['training_time'],
                           0)  # if model was trained, there should be some training time in evaluation


# test_model_variables()

if __name__ == '__main__':
    unittest.main()
