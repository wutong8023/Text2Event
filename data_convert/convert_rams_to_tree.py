import os
import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Instance:
    text: str = None
    target: str = None
    event_types: List = None
    events: List[Any] = None


@dataclass
class Event:
    event_type: str = None
    trigger: str = None
    role_and_arguments: Dict = None


def flat_sentences(sentences: List[List[str]]) -> List[str]:
    """
    flat a list of list to a list
    Args:
        sentences: List[List[str]] source sentences

    Returns:
    flatted_sentence
    """
    flatted_sentence = []
    for sentence in sentences:
        flatted_sentence += sentence
    return flatted_sentence


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

    # parse sentence
    sentences = raw_instance["sentences"]  # List[List[str]]
    flatted_sentence = flat_sentences(sentences)

    # parse event
    events = []
    event_triggers = raw_instance["evt_triggers"]
    for trigger in event_triggers:
        event_type = trigger[2][0][0]

        trigger_word = ' '.join(flatted_sentence[trigger[0]:trigger[1] + 1])

        role_and_arguments = {}
        for link in raw_instance["gold_evt_links"]:
            if trigger_word == ' '.join(flatted_sentence[link[0][0]:link[0][1] + 1]):  # assert the trigger is matched
                role = link[2]
                argument = ' '.join(flatted_sentence[link[1][0]:link[1][1] + 1])
                role_and_arguments[role] = argument
        events.append(Event(event_type=event_type, trigger=trigger_word, role_and_arguments=role_and_arguments))

    text = ' '.join(flatted_sentence)
    target = events_to_tree(events)
    event_types = [event.event_type for event in events]
    return Instance(text=text, target=target, event_types=event_types, events=events)


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


if __name__ == '__main__':
    # load data
    file_names = ["train", "dev", "test"]

    input_files = {f_name: f"./data/RAMS_1.0/data/{f_name}.jsonlines" for f_name in file_names}

    output_files = {f_name: f"./data/RAMS_1.0/data_formatted/{f_name.replace('dev', 'val')}.json" for f_name in
                    file_names}
    schema_file = f"./data/RAMS_1.0/data_formatted/event.schema"
    count_file = "./data/RAMS_1.0/data_formatted/count.csv"

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

