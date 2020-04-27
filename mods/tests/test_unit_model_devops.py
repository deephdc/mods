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

from mods import config as cfg

cfg.sequence_len = 4
cfg.steps_ahead = 3
cfg.model_delta = False
cfg.launch_tensorboard = False
cfg.MODS_DEBUG_MODE = True

import mods.models.mods_model as MODS

from mods import utils as utl
import pandas as pd
import numpy as np

debug = True


class TestModelMethods(unittest.TestCase):
    def setUp(self):
        self.model = MODS.mods_model('test')
        self.df = pd.DataFrame(
            {
                'a': [1, 3, 2, 7, 5, 4, 8, 4, 5, 3, 6, 1, 8, 4, 5, 3, 9],
                'b': [3, 2, 7, 5, 4, 8, 4, 5, 3, 6, 1, 8, 4, 5, 3, 9, 1]
            }
        )
        self.model.set_multivariate(len(self.df.keys()))

    def test_append_nan_df(self):
        m = self.model
        df_true = pd.DataFrame(
            {
                'a': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                'b': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
            }
        )
        df = m.append_nan(self.df, len(df_true))
        df_true = df_true.set_index(df[-len(df_true):].index)
        self.assertEqual(True, df[-len(df_true):].equals(df_true))

    def test_get_tsg(self):
        m = self.model
        df = self.df
        tsg = m.get_tsg(df, m.get_steps_ahead(), m.get_batch_size())
        print(utl.tsg2tsv(tsg))

    def test_train_delta(self):
        m = self.model
        m.set_model_delta(True)
        df = pd.DataFrame(
            {
                'a': [1, 2, 3, 4, 3, 2, 1, 2, 3,  4,  3,  2,  1,  2,  3,  4,  3,  2,  1,  2,  3,  4,  5,  4,  3,  2,  1],
                'b': [1, 1, 1, 2, 2, 2, 1, 1, 1,  2,  2,  2,  1,  1,  1,  2,  2,  2,  1,  1,  1,  2,  2,  2,  1,  1,  1],
                'c': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            }
        )
        # df = pd.DataFrame(
        #     {
        #         'a': [1, 2, 1, 3, 5, 2, 6, 1, 2, 1]
        #     }
        # )
        # df = pd.DataFrame(
        #     {
        #         'a': [1, 3, 5, 7,  9,  1,  3,  5,  7,  9,  1,  3,  5,  7,  9,  1,  3,  5,  7,  9,  1,  3,  5,  7,  9],
        #         'b': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 24, 26, 28, 40, 42, 44, 46, 48, 50]
        #     }
        # )
        m.train(df)
        x_test = df[:-m.get_steps_ahead()]
        y_pred = m.predict(x_test)
        y_true = df[-len(y_pred):]
        print(y_pred)


if __name__ == '__main__':
    unittest.main()
