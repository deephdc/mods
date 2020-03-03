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
from shutil import copytree, ignore_patterns
from pathlib import Path

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
            logging.info('unzip(%s): start' % path)
            unzip(path)
            logging.info('unzip(%s): done' % path)


def include_only_zip_files(dirpath, contents):
    return set(contents) - set(ignore_patterns('*.zip')(dirpath, contents))


def prepare_data(
        remote_data_dir=cfg.app_data_remote,
        local_data_dir=cfg.app_data,
        remote_models_dir=cfg.app_models_remote,
        local_models_dir=cfg.app_models
):
    """ Function to prepare data
    """
    logging.info('prepare_data(...): start')
    try:
        # copy data directory structure from remote to local
        if remote_data_dir != local_data_dir:
            logging.info('copytree(%s, %s): start' % (remote_data_dir, local_data_dir))
            copytree(remote_data_dir, local_data_dir, ignore=include_only_zip_files)
            logging.info('copytree(%s, %s): done' % (remote_data_dir, local_data_dir))
    except Exception as e:
        logging.info(e)
    
    find_n_unzip(local_data_dir, depth=1)
    
    # copy models directory structure from remote to local
    try:
        if remote_models_dir != local_models_dir:
            logging.info('copytree(%s, %s): start' % (remote_models_dir, local_models_dir))
            copytree(remote_models_dir, local_models_dir, ignore=include_only_zip_files)
            logging.info('copytree(%s, %s): done' % (remote_models_dir, local_models_dir))
    except Exception as e:
        logging.info(e)
    
    logging.info('prepare_data(...): done')
