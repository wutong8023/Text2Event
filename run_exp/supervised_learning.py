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
    for tuning_type in ["both"]:
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        for epoch in [120]:
            for is_knowledge in [True]:
                is_knowledge = "--is_knowledge" if is_knowledge and tuning_type in ["prefix", "both", "hybrid",
                                                                                    "hybridpp"] else ""
                for no_module in [False]:
                    no_module = "--no_module" if no_module else ""
                    for model in ["t5-base"]:
                        for wo_encoder_conditioning in ["--wo_encoder_conditioning"]:  # --wo_encoder_conditioning
                            for wo_decoder_conditioning in [""]:  # --wo_decoder_conditioning
                                for wo_cross_attention in [""]:  # --wo_cross_attention
                                    for wo_constrained_decoding in [""]:  # --wo_constraint_decoding
                                        for prefix_len in [20]:
                                            for data in ["oneie/oneie_33_training"]:
                                                output_dir = f"models/sl_{tuning_type}_{wo_constrained_decoding}{wo_encoder_conditioning}{wo_cross_attention}{wo_decoder_conditioning}_{no_module}{is_knowledge}_len{prefix_len}_{data.split('/')[1]}_{current_time}"
                                                cmd = f"bash run_seq2seq_verbose_prefix.bash " \
                                                      f"-d 0 " \
                                                      f"-f tree " \
                                                      f"-m {model} " \
                                                      f"--label_smoothing 0 " \
                                                      f"-l 5e-5 " \
                                                      f"--lr_scheduler linear " \
                                                      f"--warmup_steps 2000 " \
                                                      f"-b 4 " \
                                                      f"{is_knowledge+' ' if is_knowledge != '' else ''}"\
                                                      f"{wo_constrained_decoding+' ' if wo_constrained_decoding != '' else ''}"\
                                                      f"{wo_encoder_conditioning+' ' if wo_encoder_conditioning != '' else ''}" \
                                                      f"{wo_decoder_conditioning+' ' if wo_decoder_conditioning != '' else ''}" \
                                                      f"{wo_cross_attention+' ' if wo_decoder_conditioning != '' else ''}" \
                                                      f"{no_module+' ' if no_module != '' else ''}" \
                                                      f"--epoch {epoch} " \
                                                      f"--data {data} " \
                                                      f"--output_dir {output_dir} " \
                                                      f"--tuning_type {tuning_type} " \
                                                      f"--prefix_len {prefix_len} "
                                                if cmd not in cmd_list:
                                                    info_list.append(output_dir.split("/")[-1])
                                                    cmd_list.append(cmd)
    return cmd_list, info_list
