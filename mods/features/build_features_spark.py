#!/usr/bin/env python3
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
Created on Tue Jun 26 10:41:28 2018

Building features from extracted datasets for ML/DL modeling.
The script runs in the same way for Hadoop cluster and local machine

@@author: stefan dlugolinsky
@author: giangnguyen
"""

"""
Note on Sat Oct 25 10:23:41 2018

Usage example about IISAS Hadoop cluster usage
(executed from the root directory of the deep-mods-devops directory):

"""

if __name__ == "__main__":
    import argparse
    import mods.config as cfg

    parser = argparse.ArgumentParser(
        description='MODS: Feature selection',
        usage="%s -w '1 hour' -s '10 minutes' -r '2018-04-14 -- 2019-02-18' -i data/bro/logs-parquet/ -o data/bro/series/" % __file__,
        epilog='---')
    parser.add_argument('-w', '--window-duration',
                        help='Duration of the sliding window; e.g., \'1 minute\', \'30 minutes\', \'1 hour\'',
                        type=str,
                        default=cfg.window_duration,
                        required=False)
    parser.add_argument('-s', '--slide-duration',
                        help='Duration of the window\'s slide; e.g., \'1 minute\', \'30 minutes\', \'1 hour\'. If the slideDuration is not provided, the windows will be tumbling windows.',
                        type=str,
                        default=cfg.slide_duration,
                        required=False)
    parser.add_argument('-b', '--time-range-beg',
                        help='Time range begin (inclusive) in one of the formats: YYYY-MM-DD, YYYY-MM',
                        type=str,
                        default=cfg.time_range_beg,
                        required=False)
    parser.add_argument('-e', '--time-range-end',
                        help='Time range end (exclusive) in one of the formats: YYYY-MM-DD, YYYY-MM',
                        type=str,
                        default=cfg.time_range_end,
                        required=False)
    parser.add_argument('-i', '--input-dir',
                        help='Input directory of logs',
                        type=str,
                        default=cfg.dir_parquet,
                        required=False)
    parser.add_argument('-o', '--output-dir',
                        help='Output directory',
                        type=str,
                        default=cfg.app_data_features,
                        required=False)
    parser.add_argument('-f', '--format',
                        help='Output format; i.e.: parquet, tsv',
                        type=str,
                        choices=['tsv', 'parquet'],
                        default=cfg.format,
                        required=False)
    parser.add_argument('-p', '--partitions',
                        help='Number of partitions',
                        type=int,
                        default=cfg.partitions,
                        required=False)
    parser.add_argument('-F', '--features',
                        help='Features to select (conn, dns, sip, ssh, ssl, http)',
                        type=str,
                        default=cfg.features,
                        required=False)
    parser.add_argument('-n', '--regex-local-network',
                        help='Regular expression matching the local network. The regex is used to resolve the networking direction; i.e., in, out or internal.',
                        type=str,
                        default=cfg.regex_local_network,
                        required=False)
    args = parser.parse_args()
    main(args)
