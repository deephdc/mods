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
Created on Mon Oct 15 10:14:37 2018

Train models with first order differential to monitor changes

@author: giang nguyen
@author: stefan dlugolinsky
"""

import argparse
import time

import mods.config as cfg
import mods.models.api as api


# during development it might be practical
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """

    start = time.time()
    ret = ''
    if args.file is not None:
        ret = api.predict_file(args, full_paths=True)
    elif args.url is not None:
        ret = api.predict_url(args, full_paths=True)
    # elif args.data is not None:
    #     ret = api.predict_data(args)
    else:
        ret = api.get_metadata()
    print(ret)
    print("Elapsed time:  ", time.time() - start)


__data_help = """
String with data to predict on.

An, example on how to read data 
from a file into a command line
argument:

--data "`data=''; while read line; do data=$data$line$'\n'; done; echo \"$data\"`"
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model parameters')

    predict_args = api.get_predict_args()

    for key, val in predict_args.items():
        parser.add_argument('--%s' % key,
                            default=val['default'],
                            type=type(val['default']),  # may just put str
                            help=val['help'])
        print(key, val)
        print(type(val['default']))

    parser.add_argument('--file', type=str, default=cfg.data_test, help='File to do prediction on, full path')
    parser.add_argument('--url', type=str, help='URL with the data to do prediction on')
    # parser.add_argument('--data', type=str, help='String with data to do prediction on')

    args = parser.parse_args()
    print("Vars:", vars(args))

    main()
