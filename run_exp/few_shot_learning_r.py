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
    for tuning_type in ["both", "fine", "prefix"]:
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        for epoch in [120]:
            for is_knowledge in [False]:
                is_knowledge = "--is_knowledge" if is_knowledge else ""
                for no_module in [True]:
                    no_module = "--no_module" if no_module else ""
                    for model in ["t5-base"]:
                        for prefix_len in [5]:
                            for shot in [1, 2, 5, 10, 15]:
                                for data in [f"oneie/rams/{str(shot)}_shot"]:
                                    target_output_dir = f"models/fslr_{tuning_type}_{no_module}{is_knowledge}_len{prefix_len}_shot{shot}_{data.split('/')[1]}_{current_time}"
                                    cmd = f"bash run_seq2seq_verbose_prefix.bash " \
                                          f"-d 0 " \
                                          f"-f tree " \
                                          f"-m {model} " \
                                          f"--label_smoothing 0 " \
                                          f"-l 5e-5 " \
                                          f"--lr_scheduler linear " \
                                          f"--warmup_steps 2000 " \
                                          f"-b 8 " \
                                          f"{is_knowledge} " \
                                          f"{no_module} " \
                                          f"--epoch {epoch} " \
                                          f"--data {data} " \
                                          f"--prefix_len {prefix_len} "\
                                          f"--output_dir {target_output_dir} " \
                                          f"--tuning_type {tuning_type} "
                                    info_list.append(target_output_dir.split("/")[-1])
                                    cmd_list.append(cmd)
    return cmd_list, info_list
