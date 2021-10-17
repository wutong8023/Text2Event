#!/usr/bin/env python
#
# Copyright the CoLL team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Intro: 
# Author: Tongtong Wu
# Time: Oct 14, 2021
"""

import os
import time

from argparse import ArgumentParser

import run_exp.transfer_learning as exp0
import run_exp.few_shot_learning as exp1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp", default=1, type=int, help="experiment id")
    args = parser.parse_args()

    exps = [exp0, exp1]

    cmds = exps[args.exp].get_cmd()

    for cmd in cmds:
        print("line 8:", cmd)
        os.system(cmd)
        # time.sleep(5)