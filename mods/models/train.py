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
    kwargs = {k.replace('pd_', ''): v for k, v in vars(args).items()}
    m.train(kwargs)
    print("Elapsed time:  ", time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('data', type=str, help=cfg.data_train_help)
    parser.add_argument('model', type=str, help=cfg.model_name_help)
    parser.add_argument('--dir-models', type=str, default='', help='Directory, where to store trained model.')
    parser.add_argument('--dir-data', type=str, default='', help='Directory containing training data.')
    parser.add_argument('--multivariate', type=int, default=cfg.multivariate, help=cfg.multivariate_help)
    parser.add_argument('--sequence-len', type=int, default=cfg.sequence_len, help=cfg.sequence_len_help)
    parser.add_argument('--model-delta', action='store_true', help=cfg.model_delta_help)
    parser.add_argument('--interpolate', action='store_true', help=cfg.interpolate_help)
    parser.add_argument('--model-type', type=str, default=cfg.model_type, help=cfg.model_type_help)
    parser.add_argument('--n-epochs', type=int, default=cfg.n_epochs, help=cfg.n_epochs_help)
    parser.add_argument('--epochs-patience', type=int, default=cfg.epochs_patience, help=cfg.epochs_patience_help)
    parser.add_argument('--blocks', type=int, default=cfg.blocks, help=cfg.blocks_help)

    # pd - pandas
    parser.add_argument('--pd-usecols', type=str, default=cfg.usecols, help=cfg.usecols_help)
    parser.add_argument('--pd-sep', type=str, default='\t', help='')
    parser.add_argument('--pd-skiprows', type=int, default=0, help='')
    parser.add_argument('--pd-skipfooter', type=int, default=0, help='')
    parser.add_argument('--pd-engine', type=str, default='python', help='')
    parser.add_argument('--pd-header', type=str, default='0', help='')

    args = parser.parse_args()
    main()
