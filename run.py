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

import run_exp.testbed_fsl as tb_fsl
import run_exp.testbed_tl as tb_tl

import run_exp.transfer_learning as tl

import run_exp.few_shot_learning as fsl
import run_exp.few_shot_learning_r as fslr
import run_exp.few_shot_learning_w as fslw

import run_exp.zero_shot_learning as zsl
import run_exp.zero_shot_learning_r as zslr
import run_exp.zero_shot_learning_w as zslw

import run_exp.supervised_learning as sl






def batch_submit_multi_jobs(cmd_list, info_list, platform: str, split_num: int = 4, partition="g"):
    assert len(cmd_list) == len(info_list)
    
    content = []
    file_name = "./job_base_{pltf}.sh".format(pltf=platform)
    file_out = "./job_{pltf}.sh".format(pltf=platform)
    
    cmd_list_frac = []
    info_list_frac = []
    
    flag_idx = 0
    while flag_idx < len(cmd_list):
        if (flag_idx + split_num) <= len(cmd_list):
            next_flag_idx = flag_idx + split_num
        else:
            next_flag_idx = len(cmd_list)
        
        sub_cmd_list = cmd_list[flag_idx:next_flag_idx:]
        sub_info_list = info_list[flag_idx:next_flag_idx:]
        
        cmd_list_frac.append(sub_cmd_list)
        info_list_frac.append(sub_info_list)
        
        flag_idx = next_flag_idx
    
    with open(file_name) as in_file:
        for line in in_file:
            content.append(line)
    for i, sub_cmd_list in enumerate(cmd_list_frac):
        with open(file_out, "w") as out_file:
            
            # job_name
            job_name = "__".join(info_list_frac[i])
            print("- JOB NAME: ", job_name)
            if platform == "group":
                _info = "#SBATCH -J {job_name}\n".format(job_name=job_name)
                content[21] = _info
                # SBATCH -o log/fs2s-iwslt-%J.out
                # SBATCH -e log/fs2s-iwslt-%J.err
                _out_file = "#SBATCH -o log/%J-{job_name}.out\n".format(job_name=job_name)
                content[15] = _out_file
                _err_file = "#SBATCH -e log/%J-{job_name}.err\n".format(job_name=job_name)
                content[16] = _err_file
            
            else:
                _partition = "#SBATCH --partition={var}\n".format(var=partition)
                content[2] = _partition
                _info = "#SBATCH --job-name={job_name}\n".format(job_name=job_name)
                content[3] = _info
                
                # SBATCH --output=log/fs2s-iwslt-%j.out
                # SBATCH --error=log/fs2s-iwslt-%j.err
                _out_file = "#SBATCH --output=log/%j-{job_name}.out\n".format(job_name=job_name)
                content[4] = _out_file
                _err_file = "#SBATCH --error=log/%j-{job_name}.err\n".format(job_name=job_name)
                content[5] = _err_file
            
            for line in content:
                out_file.write(line)
            
            # command
            for cmd in sub_cmd_list:
                out_file.write(cmd)
                out_file.write("\n\n")
        cmd = "sbatch job_{pltf}.sh".format(pltf=platform)
        os.system(cmd)


def batch_run_interactive(cmd_list: [str], order=1):
    # print(cmd_list)
    for i in cmd_list[::order]:
        print(i)
    for i in cmd_list[::order]:
        try:
            os.system(i)
            time.sleep(10)
            print(i)
        except:
            print(i, " failed!")


# cancel slurm jobs
def batch_cancel(job_start: int, num: int, platform: str):
    for i in range(job_start, job_start + num):
        if platform == "group":
            cmd = "scancel -v {i}".format(i=i)
        else:
            cmd = "scancel {i}".format(i=i)
        os.system(cmd)


if __name__ == '__main__':
    exps = {
        "fsl": fsl,
        "fslr": fslr,
        "fslw": fslw,

        "zsl": zsl,
        "zslr": zslr,
        "zslw": zslw,
        
        
        "sl": sl,
        "tl": tl,
        
        "tb_tl": tb_tl,
        "tb_fsl": tb_fsl
    }
    
    parser = ArgumentParser()
    parser.add_argument("--exp", default="fsl", type=str, help="experiment id",
                        choices=exps.keys())
    parser.add_argument("--split", default=3, type=int, help="experiment id")
    parser.add_argument("--pltf", default="group", type=str, help="cluster: m3, group",
                        choices=["m3", "group"])
    parser.add_argument("--sbatch", action="store_true")
    parser.add_argument("--cancel", type=int)
    parser.add_argument("--range", type=int)
    args = parser.parse_args()
    
    
    cmd_list, info_list = exps[args.exp].get_cmd()
    
    if args.sbatch:
        batch_submit_multi_jobs(cmd_list, info_list, args.pltf, split_num=args.split, partition="m3g")
    else:
        # cmd: submit batch jobs for multi jobs
        # optional partition for m3: dgx , m3g, m3h, m3e
        batch_run_interactive(cmd_list, order=1)
    
    if args.cancel:
        batch_cancel(args.cancel, args.range, platform=args.pltf)
