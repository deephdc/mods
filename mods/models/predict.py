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
import mods.models.model as MODEL


# during development it might be practical
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """
    start = time.time()
    MODEL.get_model(args.model)
    ret = ''
    if args.file is not None:
        ret = MODEL.predict_file(args.file)
    elif args.url is not None:
        ret = MODEL.predict_url(args.url)
    elif args.data is not None:
        ret = MODEL.predict_data([str(args.data).encode('utf-8')])
    else:
        MODEL.get_metadata()
        return
    print(ret)
    print("Elapsed time:  ", time.time() - start)


if __name__ == '__main__':
    train_args = MODEL.get_train_args()
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('model', type=str, default=cfg.default_model, help=train_args['model_name']['help'])
    parser.add_argument('--file', type=str, help='File to do prediction on, full path')
    parser.add_argument('--url', type=str, help='URL with the data to do prediction on')
    parser.add_argument('--data', type=str,
                        help='String with data to predict on. An, example on how to read data from a file into a command line argument: --data "`data=\'\'; while read line; do data=$data$line$\'\\n\'; done; echo \\"$data\\"`" model.zip')

    args = parser.parse_args()

    main()
