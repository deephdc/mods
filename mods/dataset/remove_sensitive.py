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
Created on Fri Oct  5 09:07:33 2018

Removing sensitive information from raw logs e.g. IPs, emails.
Keep only a part of data neccesary to demonstrate proof-of-the-concept.

python remove_sensitive.py --h

@author: stefan dlugolinsky
"""


import mods.config as cfg


def main(argv):
    print('MODS-DAS: Removing sensitive information from raw logs')
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                description='MODS-DAS Removing sensitive info from logs', 
                epilog='---')
    parser.add_argument("--in",     
                        default=cfg.dir_logs,
                        dest="dir_logs", 
                        help="logs directory",
                        metavar="path/to/dir/")
    parser.add_argument("--out", 
                        default=cfg.dir_cleaned,
                        dest="dir_cleaned", 
                        help="cleaned logs directory",
                        metavar="path/to/dir/")
    args = parser.parse_args()
    main(args)
