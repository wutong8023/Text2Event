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

RECORD_LIST = ["file_name",
               "test_role_F1",
               "test_role_P",
               "test_role_R",
               "test_trigger_F1",
               "test_trigger_P",
               "test_trigger_R"]


@dataclass
class Result:
    file_name: str = None
    test_role_F1: float = -1.0
    test_role_P: float = -1.0
    test_role_R: float = -1.0
    test_trigger_F1: float = -1.0
    test_trigger_P: float = -1.0
    test_trigger_R: float = -1.0


def select_files(model_dir: str, file_prefix_name: str = None):
    file_list = []
    
    for f_path, dirs, fs in os.walk(model_dir):
        for f in fs:
            run_name = f_path.split("/")[1]
            if f == "test_results_seq2seq.txt" and "checkpoint" not in f_path:
                if file_prefix_name is None or run_name.startswith(file_prefix_name):
                    file_list.append(os.path.join(f_path, f))
    
    file_list = sorted(file_list)
    return file_list


def _parse_result_from_file(file_path: str):
    result = Result()
    result.file_name = file_path.split("/")[1]
    with open(file_path, "r") as file_in:
        for line in file_in:
            record = line.split(" = ")
            if record[0].replace("-", "_") in result.__dict__:
                setattr(result, record[0].replace("-", "_"), record[1].strip())
        pass
    return result


def parse_result_from_files(file_list: List):
    csv_content = [f"{', '.join(RECORD_LIST)},\n"]
    for file_path in file_list:
        result = _parse_result_from_file(file_path)
        result_str = format_result_to_csv(result)
        csv_content.append(result_str)
    return csv_content


def format_result_to_csv(result: Result):
    record_str = ""
    for r in RECORD_LIST:
        record_str += f"{getattr(result, r)}, "
    record_str += "\n"
    return record_str


def save_csv(result_dir: str, data: List[str], name_prefix: str = None, mode: str = "w", pltf:str="m3"):
    if name_prefix is None:
        out_file_path = os.path.join(result_dir, f"{pltf}_all.csv")
    else:
        out_file_path = os.path.join(result_dir, f"{pltf}_{name_prefix}.csv")
    
    with open(out_file_path, mode) as file_out:
        for line in data:
            file_out.write(line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--name_prefix", required=False, type=str)
    parser.add_argument("--mode", required=False, default="w", choices=["a", 'w'])
    parser.add_argument("--pltf", required=True, default="m3", choices=["m3", 'group'])
    args = parser.parse_args()
    
    models_dir = "testbed_models/"
    result_dir = "testbed_results/"
    
    file_list = select_files(models_dir, file_prefix_name=args.name_prefix)
    csv_contect = parse_result_from_files(file_list)
    save_csv(result_dir=result_dir, name_prefix=args.name_prefix, data=csv_contect, mode=args.mode, pltf=args.pltf)
