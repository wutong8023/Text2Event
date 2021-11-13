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
    cmd_template = "bash run_seq2seq_verbose_prefix.bash -d 0 -f tree -m t5-base --label_smoothing 0 -l 5e-5 --lr_scheduler linear --warmup_steps 2000 -b 16 --tuning_type prefix"
    cmd_list = []
    info_list = []
    # source_data = "oneie/oneie_23_training"
    source_data = "oneie/few-shot_23_test_10"
    # for tuning_type in ["prefix", "both", "fine", "adapter", "both_adapter"]:
    for tuning_type in ["fine"]:
        for prefix_len in [20]:
            for is_knowledge in [True, False]:
                is_knowledge = "--is_knowledge" if is_knowledge and tuning_type in ["prefix", "both"] else ""
                for no_module in [False]:
                    no_module = "--no_module" if no_module else ""
                    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
                    source_output_dir = f"./models/source_{tuning_type}_{no_module}{is_knowledge}_len{prefix_len}_{source_data.split('/')[1]}_{current_time}"
                    print(source_output_dir)
                    # training on source domain
                    for model_name in ["t5-base"]:
                        for epoch in [120]:
                            cmd = f"bash run_seq2seq_verbose_prefix.bash " \
                                  f"-d 0 " \
                                  f"-f tree " \
                                  f"-m {model_name}  " \
                                  f"-l 5e-5 " \
                                  f"--label_smoothing 0 " \
                                  f"--lr_scheduler linear " \
                                  f"--warmup_steps 2000 " \
                                  f"-b 8 " \
                                  f"{is_knowledge} " \
                                  f"{no_module} " \
                                  f"--epoch {epoch} " \
                                  f"--data {source_data} " \
                                  f"--prefix_len {prefix_len} " \
                                  f"--output_dir {source_output_dir} " \
                                  f"--tuning_type {tuning_type} "
                            if cmd not in cmd_list:
                                info_list.append(source_output_dir.split("/")[-1])
                                cmd_list.append(cmd)
                    # transfer learning
                    """
                    models/CF_${date}-${time}_${model_name_log}_${tuning_type}_${data_name}
                    """
                    # for model_name in [f"models_trained/CF_{date}_{tuning_type}"]:
                    
                    for epoch in [120]:
                        for shot in [1, 2, 5]:
                            for data in [f"oneie/oneie_{str(shot)}_ft"]:
                                target_output_dir = f"./models/target_{tuning_type}_{no_module}{is_knowledge}_len{prefix_len}_shot{shot}_{data.split('/')[1]}_{current_time}__sourcedata-{source_data.split('/')[1]}_{current_time}"
                                print(target_output_dir)
                                cmd = f"bash run_seq2seq_verbose_prefix.bash " \
                                      f"-d 0 " \
                                      f"-f tree " \
                                      f"-m {source_output_dir} " \
                                      f"--label_smoothing 0 " \
                                      f"-l 5e-5 " \
                                      f"--lr_scheduler linear " \
                                      f"--warmup_steps 2000 " \
                                      f"-b 8 " \
                                      f"{is_knowledge} " \
                                      f"{no_module} " \
                                      f"--epoch {epoch} " \
                                      f"--data {data} " \
                                      f"--prefix_len {prefix_len} " \
                                      f"--output_dir {target_output_dir} " \
                                      f"--tuning_type {tuning_type} "
                                if cmd not in cmd_list:
                                    info_list.append(target_output_dir.split("/")[-1])
                                    cmd_list.append(cmd)
    return cmd_list, info_list
