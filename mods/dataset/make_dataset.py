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

@author: stefan dlugolinsky
"""


import logging
import os
import zipfile
from pathlib import Path

from distutils.dir_util import copy_tree
from dotenv import find_dotenv, load_dotenv

import mods.config as cfg


def unzip(file, dst_dir=None):
    if not dst_dir:
        dst_dir = os.path.dirname(os.path.abspath(file))
    
    # extract all
    # zip_ref = zipfile.ZipFile(file, 'r')
    # zip_ref.extractall(dst_dir)
    # zip_ref.close()
    
    # extract non-existing
    with zipfile.ZipFile(file) as zip_file:
        for member in zip_file.namelist():
            ff = os.path.join(dst_dir, member)
            if not (os.path.exists(ff) and os.path.isfile(ff)):
                zip_file.extract(member, dst_dir)


def find_n_unzip(dir, depth=0):
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        if depth != 0 and os.path.isdir(path):
            find_n_unzip(path, depth - 1)
        elif os.path.isfile(path) and path.lower().endswith('zip'):
            logging.info('unzipping: %s' % path)
            unzip(path)


def prepare_data(
        remote_data_dir=cfg.app_data_remote,
        local_data_dir=cfg.app_data,
        remote_models_dir=cfg.app_models_remote,
        local_models_dir=cfg.app_models
):
    """ Function to prepare data
    """
    # copy data directory structure from remote to local
    logging.info('copy_tree(%s, %s): start' % (remote_data_dir, local_data_dir))
    copy_tree(remote_data_dir, local_data_dir)
    logging.info('copy_tree(%s, %s): done' % (remote_data_dir, local_data_dir))
    
    # find and unzip data
    find_n_unzip(local_data_dir, depth=1)
    
    # copy models directory structure from remote to local
    logging.info('copy_tree(%s, %s): start' % (remote_models_dir, local_models_dir))
    copy_tree(remote_models_dir, local_models_dir)
    logging.info('copy_tree(%s, %s): done' % (remote_models_dir, local_models_dir))


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
