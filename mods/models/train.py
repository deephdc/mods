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
import time

# import project config.py
import mods.config as cfg
import mods.models.model as m


# during development it might be practical
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """
    start = time.time()
    m.train(args)
    print("Elapsed time:  ", time.time() - start)


if __name__ == '__main__':
    train_args = m.get_train_args()
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('data', type=str, default=cfg.data_train, help='Training data.')
    parser.add_argument('model', type=str, help=train_args['model_name']['help'])
    parser.add_argument('--dir-models', type=str, default='', help='Directory, where to store trained model.')
    parser.add_argument('--dir-data', type=str, default='', help='Directory containing training data.')
    parser.add_argument('--multivariate', type=int, default=cfg.multivariate, help=train_args['multivariate']['help'])
    parser.add_argument('--sequence-len', type=int, default=cfg.sequence_len, help=train_args['sequence_len']['help'])
    parser.add_argument('--model-delta', action='store_true', help=train_args['model_delta']['help'])
    parser.add_argument('--interpolate', action='store_true', help=train_args['interpolate']['help'])
    parser.add_argument('--model-type', type=str, default=cfg.model_type, help=train_args['model_type']['help'])
    parser.add_argument('--n-epochs', type=int, default=cfg.n_epochs, help=train_args['n_epochs']['help'])
    parser.add_argument('--epochs-patience', type=int, default=cfg.epochs_patience,
                        help=train_args['epochs_patience']['help'])
    parser.add_argument('--blocks', type=int, default=cfg.blocks, help=train_args['blocks']['help'])
    parser.add_argument('--usecols', type=str, default=cfg.usecols, help=train_args['usecols']['help'])

    args = parser.parse_args()

    main()
