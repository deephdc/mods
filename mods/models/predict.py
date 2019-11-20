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

import mods.models.api_v2 as api
from mods.models.api_v2 import PredictArgsSchema


# during development it might be practical
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """
    start = time.time()
    kwargs = vars(args)
    # kwargs['full_paths'] = str(True)
    if 'time_ranges_excluded' in kwargs.keys() and isinstance(kwargs['time_ranges_excluded'], str):
        kwargs['time_ranges_excluded'] = kwargs['time_ranges_excluded'].split(';')
    ret = api.predict(**kwargs)
    end = time.time()
    print("Elapsed time:  ", end - start)
    print(ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')
    for field_name, field in PredictArgsSchema().fields.items():
        parser.add_argument('--%s' % field_name, default=field.missing, required=field.required)
        print(field_name, field)
    args = parser.parse_args()
    main()
