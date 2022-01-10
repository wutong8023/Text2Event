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
# Time: , 2022
"""

import os
import json
import random

source_files = {
    "ar": "../data/text2tree/ace05-AR/data_formatted",
    "zh": "../data/text2tree/ace05-ZH/data_formatted",
    "en": "../data/text2tree/oneie/oneie_33_training"
}

files = {
    "train": "train.json",
    "test": "test.json",
    "val": "val.json",
    "schema": "event.schema"
}


def move_file(source, target, file_name):
    os.system(f"cp {os.path.join(source, file_name)} {target}")


def merge_files(file1, file2, target_file):
    data = []
    with open(file1) as file_in1:
        for line in file_in1:
            data.append(line)
    with open(file2) as file_in2:
        for line in file_in2:
            data.append(line)
    
    with open(target_file, "w") as file_out:
        for line in data:
            file_out.write(line)


if __name__ == '__main__':
    target_files = [
        # "../data/text2tree/multi-lingual/EN_ZH_SL",
        # "../data/text2tree/multi-lingual/EN+ZH_ZH_SL",
        # "../data/text2tree/multi-lingual/EN+ZH_EN_SL",
        # "../data/text2tree/multi-lingual/ZH_EN_SL",
        "../data/text2tree/multi-lingual/EN_AR_SL",
        "../data/text2tree/multi-lingual/EN+AR_AR_SL",
        "../data/text2tree/multi-lingual/EN+AR_EN_SL",
        "../data/text2tree/multi-lingual/AR_EN_SL"
    ]
    
    for folder in target_files:
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    for i, f in enumerate(["test", "val"]):
        language = "en"
        move = f
        tgt_id = 3
        move_file(source_files[language], target_files[tgt_id], files[move])

    
    # merge_files(file1=os.path.join(source_files["en"], files["train"]),
    #             file2=os.path.join(source_files["ar"], files["train"]),
    #             target_file=os.path.join(target_files[2], files["train"]))
