import os
import random
import re
import json
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Instance:
    text: str = None
    target: str = None
    event_types: List = None
    events: List[Any] = None
    instance_id: str = None

    def __hash__(self):
        return hash(self.instance_id)


@dataclass
class Event:
    event_type: str = None
    trigger: str = None
    role_and_arguments: Dict = None


def events_to_tree(events: List[Event]) -> str:
    """
    get a sequence of events, for example "[event_type trigger [role1 argument1] [role2 argument2]]".
    Args:
        events: List[Event]

    Returns:
    events_tree: str
    """
    extra_ids = ["<extra_id_0>", "<extra_id_1>"]

    trees = []
    for event in events:
        event_type_name = event.event_type.split(".")[1]
        pairs = []
        for role in event.role_and_arguments.keys():
            role_name = re.split('\d+', role)[-1]
            pair = f"{extra_ids[0]} {role_name} {event.role_and_arguments[role]} {extra_ids[1]}"
            pairs.append(pair)
        tree = f"{extra_ids[0]} {event_type_name} {event.trigger} {' '.join(pairs)} {extra_ids[1]}"
        trees.append(tree)

    events_tree = f"{extra_ids[0]} {' '.join(trees)} {extra_ids[1]}"
    return events_tree


def parse_instance(line: str) -> Instance:
    """
    parse raw data into a Instance
    Args:
        line:

    Returns:
    instance: Instance
    """
    raw_instance = json.loads(line)

    # parse event
    events = []
    event_mentions = raw_instance["event_mentions"]
    for mention in event_mentions:
        event_type = mention["event_type"]
        trigger_word = mention["trigger"]["text"]

        role_and_arguments = {}

        for argument in mention["arguments"]:
            role_and_arguments[argument["role"]] = argument["text"]
        events.append(Event(event_type=event_type, trigger=trigger_word, role_and_arguments=role_and_arguments))

    instance_id = raw_instance["doc_id"]
    text = raw_instance["text"]
    target = events_to_tree(events)
    event_types = [event.event_type for event in events]
    return Instance(text=text, target=target, event_types=event_types, events=events, instance_id=instance_id)


def format_instance(instance: Instance) -> str:
    """
    get the seq2seq instance from Instance
    Args:
        instance: Instance

    Returns:
    formatted_instance: str
    """
    formatted_instance = {
        "text": instance.text,
        "event": instance.target
    }
    formatted_instance = json.dumps(formatted_instance)
    return formatted_instance


def get_schema(instances: List[Instance], event_role_pair: Dict):
    for instance in instances:
        for event in instance.events:
            event_type = event.event_type.split(".")[1]
            if event_type not in event_role_pair.keys():
                event_role_pair[event_type] = set()
            for role in event.role_and_arguments.keys():
                role = re.split("\d+", role)[-1]
                event_role_pair[event_type].add(role)
    return event_role_pair


def process_fsl(data: List[Instance], num_train: int, num_test: int, num_dev: int):
    grouped_data = {}
    for instance in data:
        # print("instance type", type(instance))
        for event_type in instance.event_types:
            if event_type not in grouped_data.keys():
                grouped_data[event_type] = [instance]
            else:
                grouped_data[event_type].append(instance)

    sorted_data = {key: grouped_data[key] for key, value in sorted(grouped_data.items(), key=lambda item: len(item[1]))}
    print({k: len(sorted_data[k]) for k in sorted_data.keys()})

    train_set, test_set, dev_set = set(), set(), set()
    seen_set = set()
    train_data, test_data, dev_data = [], [], []
    for event_type in sorted_data.keys():
        temp_test_num = min(num_test, int(len(sorted_data[event_type]) / 4))
        temp_dev_num = min(num_dev, int(len(sorted_data[event_type]) / 4))
        temp_trn_num = min(num_train, len(sorted_data[event_type]))
        count_trn, count_test, count_dev = 0, 0, 0
        for inst in sorted_data[event_type]:
            inst_id = inst.instance_id
            if inst_id not in train_set and inst_id not in dev_set and count_test < temp_test_num:
                test_set.add(inst_id)
                if inst_id not in seen_set:
                    seen_set.add(inst_id)
                    test_data.append(inst)
                count_test += 1
                continue
            if inst_id not in train_set and inst_id not in test_set and count_dev < temp_dev_num:
                dev_set.add(inst_id)
                if inst_id not in seen_set:
                    seen_set.add(inst_id)
                    dev_data.append(inst)
                count_dev += 1
                continue
            if inst_id not in test_set and inst_id not in dev_set and count_trn < temp_trn_num:
                train_set.add(inst_id)
                if inst_id not in seen_set:
                    seen_set.add(inst_id)
                    train_data.append(inst)
                count_trn += 1
                continue

    return train_data, dev_data, test_data


def process_zsl(data: List[Instance]):
    grouped_data = {}
    for instance in data:
        # print("instance type", type(instance))
        for event_type in instance.event_types:
            if event_type not in grouped_data.keys():
                grouped_data[event_type] = [instance]
            else:
                grouped_data[event_type].append(instance)

    sorted_data = {key: grouped_data[key] for key, value in sorted(grouped_data.items(), key=lambda item: len(item[1]))}
    print({k: len(sorted_data[k]) for k in sorted_data.keys()})

    train_set, test_set, dev_set = set(), set(), set()
    labels = [key for key in sorted_data.keys()]
    for label in labels[13:23]:
        for instance in sorted_data[label]:
            test_set.add(instance)

    for label in labels[:13] + labels[23:]:
        dev_num = min(20, int(len(sorted_data[label]) / 4))
        count_dev = 0
        for instance in sorted_data[label]:
            if instance not in test_set and instance not in train_set and count_dev < dev_num:
                dev_set.add(instance)
                count_dev += 1
            else:
                train_set.add(instance)
    return list(train_set), list(dev_set), list(test_set)


def process_tl(data: List[Instance], num_target_train: List, mini_num_test: int):
    """

    Args:
        data:
        num_target_train:
        mini_num_test:

    Returns:
    source_train_list: List[Instance],
    dev_list: List[Instance],
    test_list: List[Instance],
    num_shot_train: Dict[List[Instance]]

    """
    target_tasks = []  # [train_set, dev_set, test_set]

    grouped_data = {}
    for instance in data:
        # print("instance type", type(instance))
        for event_type in instance.event_types:
            if event_type not in grouped_data.keys():
                grouped_data[event_type] = [instance]
            else:
                grouped_data[event_type].append(instance)

    sorted_data = {key: grouped_data[key] for key, value in sorted(grouped_data.items(), key=lambda item: len(item[1]))}
    print({k: len(sorted_data[k]) for k in sorted_data.keys()})

    target_set, final_target_dev_test = set(), set()

    labels = [key for key in sorted_data.keys()]

    # process target
    for label in labels[13:23]:
        for inst in sorted_data[label]:
            target_set.add(inst)
        print("target set", len(target_set))

    candidate_set = target_set
    num_shot_train = {}

    for num_shot in sorted(num_train)[::-1]:
        target_train_set = set()
        target_dev_test = set()
        bin_per_class = {label: [] for label in labels[13:23]}

        for inst in candidate_set:
            assert type(inst) is Instance
            entity_types = inst.event_types

            # add instance only when needed
            check_num = [len(bin_per_class[i]) < num_shot for i in bin_per_class.keys() if i in entity_types]
            if True in check_num:
                # do add instance to train
                for entity_type in entity_types:
                    if entity_type in bin_per_class.keys():
                        bin_per_class[entity_type].append(inst)
                target_train_set.add(inst)
            else:
                # do eval / test
                target_dev_test.add(inst)
                pass

        num_shot_train[num_shot] = list(target_train_set)
        candidate_set = target_train_set
        if len(final_target_dev_test) == 0:
            final_target_dev_test = target_dev_test
    pass

    # split dev test
    dev_test_size = len(final_target_dev_test)
    dev_test_list = list(final_target_dev_test)

    print("final target_test_set:", dev_test_size)
    index = np.random.choice(np.arange(0, dev_test_size), int(0.9 * dev_test_size))
    print(f"dev_test_size:{dev_test_size}, test_size: {len(index)}")

    test_list, dev_list = [], []
    for i, inst in enumerate(dev_test_list):
        if random.random() > 0.7:
            test_list.append(inst)
        else:
            dev_list.append(inst)

    source_train_set = set()
    for label in labels[:13] + labels[23:]:
        for inst in sorted_data[label]:
            if inst not in target_set:
                source_train_set.add(inst)
    source_train_list = list(source_train_set)

    return source_train_list, dev_list, test_list, num_shot_train


if __name__ == '__main__':
    # load data
    file_names = ["train", "dev", "test"]

    input_files = {f_name: f"./data/wikievents/{f_name}.jsonl" for f_name in file_names}

    output_files = {f_name: f"./data/wikievents/data_formatted/{f_name.replace('dev', 'val')}.json" for f_name in
                    file_names}
    schema_file = f"./data/wikievents/data_formatted/event.schema"
    count_file = "./data/wikievents/data_formatted/count.csv"

    event_role_pair = {}

    data_count = {}
    num_event_count = {}

    # parse file
    for file_name in file_names:
        file_path = input_files[file_name]

        data = []
        with open(file_path, "r") as file_in:
            for line in file_in:
                instance = parse_instance(line)
                data.append(instance)

                num_event = len(instance.events)
                if num_event in num_event_count:
                    num_event_count[num_event] += 1
                else:
                    num_event_count[num_event] = 1

                for event_type in instance.event_types:
                    event_type = ".".join(event_type.split(".")[:2])
                    if event_type not in data_count.keys():
                        data_count[event_type] = 1
                    else:
                        data_count[event_type] += 1

        output_file_path = output_files[file_name]
        with open(output_file_path, "w") as file_out:
            for instance in data:
                formatted_instance = format_instance(instance)
                file_out.write(formatted_instance + "\n")

        event_role_pair = get_schema(instances=data, event_role_pair=event_role_pair)

    events_schema = json.dumps(list(event_role_pair.keys()))
    roles = set()
    for event_type in event_role_pair.keys():
        for role in event_role_pair[event_type]:
            roles.add(role)
    roles_schema = json.dumps(list(roles))
    event_role_schema = json.dumps({event: list(roles) for event, roles in event_role_pair.items()})
    schemas = [events_schema, roles_schema, event_role_schema]

    with open(schema_file, "w") as file_out:
        file_out.write("\n".join(schemas))

    sorted_count = {key: value for key, value in sorted(data_count.items(), key=lambda item: item[1])}
    print(sorted_count)

    with open(count_file, 'w') as file_out:
        for key, value in sorted_count.items():
            file_out.write(f"{key},{value},\n")

    print(num_event_count)

    # fsl processing
    raw_data = []
    for file_name in file_names:
        file_path = input_files[file_name]
        with open(file_path, "r") as file_in:
            for line in file_in:
                instance = parse_instance(line)
                raw_data.append(instance)

    for num_train in [1, 2, 5, 10, 15]:
        data = process_fsl(raw_data, num_train=num_train, num_test=10, num_dev=10)
        output_files = {f_name: f"./data/wikievents/data_formatted/{num_train}_shot/{f_name.replace('dev', 'val')}.json"
                        for f_name in file_names}
        schema_file = f"./data/wikievents/data_formatted/{num_train}_shot/event.schema"
        if not os.path.exists(f"./data/wikievents/data_formatted/{num_train}_shot/"):
            os.mkdir(f"./data/wikievents/data_formatted/{num_train}_shot/")

        for i, file_name in enumerate(output_files):
            with open(output_files[file_name], "w") as file_out:
                for instance in data[i]:
                    formatted_instance = format_instance(instance)
                    file_out.write(formatted_instance + "\n")

        with open(schema_file, "w") as file_out:
            file_out.write("\n".join(schemas))

    # process zero_shot_learning
    data = process_zsl(raw_data)
    output_files = {f_name: f"./data/wikievents/data_formatted/zsl/{f_name.replace('dev', 'val')}.json"
                    for f_name in file_names}
    schema_file = f"./data/wikievents/data_formatted/zsl/event.schema"
    if not os.path.exists(f"./data/wikievents/data_formatted/zsl/"):
        os.mkdir(f"./data/wikievents/data_formatted/zsl/")

    for i, file_name in enumerate(output_files):
        with open(output_files[file_name], "w") as file_out:
            for instance in data[i]:
                formatted_instance = format_instance(instance)
                file_out.write(formatted_instance + "\n")

    with open(schema_file, "w") as file_out:
        file_out.write("\n".join(schemas))

    # process transfer learning
    num_train = [1, 2, 5]
    train, dev, test, shot_train_dict = process_tl(raw_data, num_train, mini_num_test=10)
    output_dir = f"./data/wikievents/data_formatted/zsl/"
    output_files = {f_name: f"{output_dir}{f_name.replace('dev', 'val')}.json"
                    for f_name in file_names}
    schema_file = f"{output_dir}event.schema"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i, file_name in enumerate(output_files):
        with open(output_files[file_name], "w") as file_out:
            for instance in [train, dev, test][i]:
                formatted_instance = format_instance(instance)
                file_out.write(formatted_instance + "\n")

    with open(schema_file, "w") as file_out:
        file_out.write("\n".join(schemas))

    for i, shot_num in enumerate(num_train):
        train = list(shot_train_dict[shot_num])
        dev = dev
        test = test

        output_dir = f"./data/wikievents/data_formatted/tl_target_{shot_num}/"
        output_files = {f_name: f"{output_dir}{f_name.replace('dev', 'val')}.json"
                        for f_name in file_names}
        schema_file = f"{output_dir}event.schema"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for i, file_name in enumerate(output_files):
            with open(output_files[file_name], "w") as file_out:
                for instance in [train, dev, test][i]:
                    formatted_instance = format_instance(instance)
                    file_out.write(formatted_instance + "\n")

        with open(schema_file, "w") as file_out:
            file_out.write("\n".join(schemas))
