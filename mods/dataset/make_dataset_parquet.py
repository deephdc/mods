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
Created on Thu May 30 08:54:16 2018

MODS: Recusively convert compressed Bro logs (gzip) into parquet format
      in rsync style

python make_dataset_parquet.py --h

@author: giangnguyen
"""

import mods.config as cfg

import gzip
import os
import shutil

from bat.log_to_parquet import log_to_parquet
from itertools import islice


# Parquet column name without dot
def replace_header_lines(file_opened, N=cfg.log_header_lines):
    header = [x.decode('utf-8').replace('.', '_').encode('utf-8')
        for x in islice(file_opened, N)]
    return header


# Decompressing gzip file
def decompressing(ffn, fn_log):
    print('\t ... decompressing to ' + fn_log)
    with open(fn_log, 'wb') as fout:
        with gzip.open(ffn, 'rb') as fin:
            header = replace_header_lines(fin)
            fout.writelines(header)
            shutil.copyfileobj(fin, fout)
    return


# parquet filenames without char ':'
# find /path/to/dir/ -type f -exec rename 's/[:]/_/g' '{}' \;
def gzip_to_parquet(dir_logs, dir_parquet):
    print(dir_logs, dir_parquet)
    fn_excluded = ['current', 'loaded_scripts']

    for root, directories, filenames in os.walk(dir_logs):
        
        for subdir in directories:
        
            subdir_parquet = (os.path.join(root, subdir)
                                .replace(dir_logs, dir_parquet) )
            
            if not os.path.exists(subdir_parquet):
                os.makedirs(subdir_parquet)
                print('mkdir ' + subdir_parquet)

        for fn in filenames:
            ffn = os.path.join(root, fn)
            
            # Parquet filename without : char
            fn_log = ( ffn.replace(dir_logs, dir_parquet)
                            .replace('.log.gz', '.log')
                            .replace(':', '_') )    
            fn_parquet = fn_log.replace('.log', '.parquet')
            print(ffn)

            if (not os.path.isfile(fn_parquet) 
                    and not any(ss in fn for ss in fn_excluded) ):
                
                decompressing(ffn, fn_log)
                
                if 'summary' not in fn_log:
                    log_to_parquet(fn_log, fn_parquet)
                    print('\t ... converted to ' + fn_parquet)
                    os.remove(fn_log)
    return


def main(argv):
    gzip_to_parquet(argv.dir_logs, argv.dir_parquet)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                description='MODS: Transform Bro logs (gzip) to parquet', 
                epilog='---')
    parser.add_argument("--in",     
                        default=cfg.dir_logs,
                        dest="dir_logs", 
                        help="logs directory",
                        metavar="path/to/dir/")
    parser.add_argument("--out", 
                        default=cfg.dir_parquet,
                        dest="dir_parquet", 
                        help="parquet directory",
                        metavar="path/to/dir/")
    args = parser.parse_args()
    main(args)
