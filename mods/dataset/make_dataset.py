# -*- coding: utf-8 -*-
# import project config.py
"""
"""
import logging
import os
import zipfile
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

import mods.config as cfg
import mods.dataset.data_utils as dutils


def unzip(file, dst_dir=None, overwrite=False):
    if not dst_dir:
        dst_dir = os.path.dirname(os.path.abspath(file))
    with zipfile.ZipFile(file) as zip_file:
        for member in zip_file.namelist():
            ff = os.path.join(dst_dir, member)
            if overwrite or not (os.path.exists(ff) and os.path.isfile(ff)):
                zip_file.extract(member, dst_dir)


def find_n_unzip(dir, depth=0, overwrite=False):
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        if depth != 0 and os.path.isdir(path):
            find_n_unzip(path, depth - 1, overwrite)
        elif os.path.isfile(path) and path.lower().endswith('zip'):
            print('unzipping: %s' % path)
            unzip(path)


def download(remote_path, local_dir):
    out, err = dutils.rclone_call(
        src_path=remote_path,
        dest_dir=local_dir,
        cmd='copy'
    )
    print('rclone_copy(%s, %s):\nout: %s\nerr: %s' % (remote_path, local_dir, out, err))


def download_n_unzip(remote_path, local_dir, depth=1, force_unzip=False):
    download(remote_path, local_dir)
    find_n_unzip(local_dir, depth=depth, overwrite=force_unzip)


def prepare_data(
        remote_data_dir=cfg.app_data_remote,
        local_data_dir=cfg.app_data,
        force_unzip=False
):
    """ Downloads data from remote storage and unzips all archives found in depth=1
    """
    download_n_unzip(remote_data_dir, local_data_dir, depth=1, force_unzip=force_unzip)


def prepare_models(
    remote_models_dir=cfg.app_models_remote,
    local_models_dir=cfg.app_models
):
    """ Downloads models from the remote storage
    """
    download(remote_models_dir, local_models_dir)


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
