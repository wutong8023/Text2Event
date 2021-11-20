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
    source_data = "oneie/zsl_oov/oneie_23_training"
    # source_data = "oneie/zsl_oov/few-shot_23_test_10"
    # for tuning_type in ["prefix", "both", "fine", "adapter", "both_adapter"]:
    for tuning_type in ["prefix", "both", "fine"]:
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        for epoch in [120]:
            for is_knowledge in [True]:
                if is_knowledge and tuning_type in ["prefix", "both", "hybrid",
                                                    "hybridpp"]:
                    is_knowledge = "--is_knowledge"
                for no_module in [False]:
                    no_module = "--no_module" if no_module else ""
                    for model in ["t5-base"]:
                        for prefix_len in [20]:
                            for data in [source_data]:
                                output_dir = f"models/source_oov_{tuning_type}_{no_module}{is_knowledge}_len{prefix_len}_{data.split('/')[1]}_{current_time}"
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
                                      f"--prefix_len {prefix_len} " \
                                      f"{no_module} " \
                                      f"--epoch {epoch} " \
                                      f"--data {data} " \
                                      f"--output_dir {output_dir} " \
                                      f"--tuning_type {tuning_type} "
                                if cmd not in cmd_list:
                                    info_list.append(output_dir.split("/")[-1])
                                    cmd_list.append(cmd)
                                
                                pass
                                # transfer learning
                                """
                                models/CF_${date}-${time}_${model_name_log}_${tuning_type}_${data_name}
                                """
                                # for model_name in [f"models_trained/CF_{date}_{tuning_type}"]:

                                for epoch in [1]:
                                    for num_schema in [0, 5, 10]:
                                        for data in [f"oneie/zsl_oov/oneie_1_ft_{str(num_schema)}"]:
                                            target_output_dir = f"./models/target_oov_{num_schema}_{tuning_type}_{no_module}{is_knowledge}_len{prefix_len}_numschema{num_schema}_{data.split('/')[1]}_{current_time}__sourcedata-{source_data.split('/')[1]}_{current_time}"
                                            print(target_output_dir)
                                            cmd = f"bash run_seq2seq_verbose_prefix.bash " \
                                                  f"-d 0 " \
                                                  f"-f tree " \
                                                  f"--no_train " \
                                                  f"-m {output_dir} " \
                                                  f"--label_smoothing 0 " \
                                                  f"-l 5e-5 " \
                                                  f"--lr_scheduler linear " \
                                                  f"--warmup_steps 2000 " \
                                                  f"-b 4 " \
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
