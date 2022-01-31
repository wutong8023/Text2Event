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
# Time: Oct 17, 2021
"""
import os

from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List

if __name__ == '__main__':
    files_contain = [
        "mono_sl_ar",
        "mono_sl_zh",
        "EN_ZH",
        "EN+ZH_ZH",
        "EN+ZH_EN",
        "ZH_EN",
        "EN_AR",
        "EN+AR_AR",
        "EN+AR_EN",
        "AR_EN"
    ]
    
    for contain in files_contain:
        pltf = "m3l"
        cmd = f"python analyze_results.py --pltf {pltf} --name_contain {contain}"
        os.system(cmd)