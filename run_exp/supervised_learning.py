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
import sys
from datetime import datetime


def get_cmd():
    cmd_list = []
    info_list = []
    for tuning_type in ["prefix", "both", "fine"]:
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        for epoch in [60]:
            for model in ["t5-base"]:
                for data in ["oneie/oneie_33_training"]:
                    target_output_dir = f"models/fsl_{tuning_type}_{data.split('/')[1]}_{current_time}"
                    cmd = f"bash run_seq2seq_verbose_prefix.bash " \
                          f"-d 0 " \
                          f"-f tree " \
                          f"-m {model} " \
                          f"--label_smoothing 0 " \
                          f"-l 5e-5 " \
                          f"--lr_scheduler linear " \
                          f"--warmup_steps 2000 " \
                          f"-b 16 " \
                          f"--epoch {epoch} " \
                          f"--data {data} " \
                          f"--output_dir {target_output_dir} " \
                          f"--tuning_type {tuning_type} "
                    info_list.append(f"sl_{tuning_type}")
                    cmd_list.append(cmd)
    return cmd_list, info_list
