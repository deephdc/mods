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


import os
import subprocess


def rclone_call(src_path, dest_dir, cmd='copy', get_output=False):
    """ Function
        rclone calls
    """
    if cmd == 'copy':
        command = (['rclone', 'copy', '--progress', src_path, dest_dir])
    elif cmd == 'ls':
        command = (['rclone', 'ls', '-L', src_path])
    elif cmd == 'check':
        command = (['rclone', 'check', src_path, dest_dir])

    if get_output:
        result = subprocess.Popen(command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
    else:
        result = subprocess.Popen(command, stderr=subprocess.PIPE)
    output, error = result.communicate()
    return output, error


def rclone_copy(src_path, dest_dir, src_type='file', verbose=False):
    """ Function for rclone call to copy data (sync?)
    :param src_path: full path to source (file or directory)
    :param dest_dir: full path to destination directory (not file!)
    :param src_type: if source is file (default) or directory
    :return: if destination was downloaded, and possible error
    """

    error_out = None

    if src_type == 'file':
        src_dir = os.path.dirname(src_path)
        dest_file = src_path.split('/')[-1]
        dest_path = os.path.join(dest_dir, dest_file)
    else:
        src_dir = src_path
        dest_path = dest_dir

    # check first if we find src_path
    output, error = rclone_call(src_path, dest_dir, cmd='ls')
    if error:
        print('[ERROR] %s (src):\n%s' % (src_path, error))
        error_out = error
        dest_exist = False
    else:
        # if src_path exists, copy it
        output, error = rclone_call(src_path, dest_dir, cmd='copy')
        if not error:
            output, error = rclone_call(dest_path, dest_dir,
                                        cmd='ls', get_output=True)
            file_size = [elem for elem in str(output).split(' ') if elem.isdigit()][0]
            print('[INFO] Copied to %s %s bytes' % (dest_path, file_size))
            dest_exist = True
            if verbose:
                # compare two directories, if copied file appears in output
                # as not found or not matching -> Error
                print('[INFO] File %s copied. Check if (src) and (dest) really match..' % (dest_file))
                output, error = rclone_call(src_dir, dest_dir, cmd='check')
                if 'ERROR : ' + dest_file in error:
                    print('[ERROR] %s (src) and %s (dest) do not match!'
                          % (src_path, dest_path))
                    error_out = 'Copy failed: ' + src_path + ' (src) and ' + \
                                dest_path + ' (dest) do not match'
                    dest_exist = False
        else:
            print('[ERROR] %s (src):\n%s' % (dest_path, error))
            error_out = error
            dest_exist = False

    return dest_exist, error_out
