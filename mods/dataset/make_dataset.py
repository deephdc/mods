# -*- coding: utf-8 -*-
# import project config.py
"""
"""
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

import mods.dataset.data_utils as dutils


def prepare_data():
    """ Function to prepare data
    """

    features_dir = 'data/features'
    features_file = 'features-20180414-20181015-win-1_hour-slide-10_minutes.tsv'

    status_feature_set, _ = dutils.maybe_download_data(
        data_dir=features_dir,
        data_file=features_file
    )

    if status_feature_set:
        print("[INFO] %s, %s  exists" % (features_dir, features_file))

    test_dir = 'data/test'
    test_file = 'w1h-s10m.tsv'

    status_test_set, _ = dutils.maybe_download_data(
        data_dir=test_dir,
        data_file=test_file
    )

    if status_test_set:
        print("[INFO] %s, %s  exists" % (test_dir, test_file))


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
