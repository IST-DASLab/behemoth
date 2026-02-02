'''
This module contains utilities to create custom tokenizers for each dataset. The
total number of tokens depends on the number of subjects, relationships, and objects
in the dataset.

All words have a single meaning, and the token space is partitioned fully between subjects,
relationships, objects, and grammatical characters.
'''

import copy
import json
import math
import os
import pickle as pkl
import shutil


EOS_TOKEN = "<|endoftext|>"
PADDING_TOKEN = "<|padding|>"
SUBJECT_TOKEN = "SS"
RELATIONSHIP_TOKEN = "RR"
OBJECT_TOKEN = "OO"
Q_TOKEN = "QQ"
A_TOKEN = "AA"

SPECIAL_TOKENS = [
    EOS_TOKEN,
    PADDING_TOKEN,
    SUBJECT_TOKEN,
    RELATIONSHIP_TOKEN,
    OBJECT_TOKEN,
    Q_TOKEN,
    A_TOKEN,
]



# We create a tokenizer based on the one for Pythia models, so we take
# the boilerplate from there.
BASE_TOKENIZER_PATH = "base_tokenizer"

try:
    from . import phrase_creation_utils as pc_lib
except:
    import phrase_creation_utils as pc_lib


def count_special_tokens():
    '''
    Counts all special (grammar) tokens, so that we know from where we can start
    assigning tokens to the various subjects, relationships, and objects.
    '''
    # TODO: This code replicates a lot of the work of the tokenizer creation. Dedupe?
    tokenizer_path = os.path.join(BASE_TOKENIZER_PATH, "tokenizer.json")

    with open(tokenizer_path, "r") as f:
        old_tokenizer = json.load(f)
    config = copy.deepcopy(old_tokenizer)
    # Remap the IDs of special characters
    special_vocab = {}
    special_vocab[EOS_TOKEN] = 0 
    special_vocab["<|padding|>"] = 1
    special_vocab["."] = 2
    next_token = 3
    for token in SPECIAL_TOKENS[2:]:
        special_vocab[chr(288) + token] = next_token
        next_token += 1
    for at in config["added_tokens"]:
        if at["id"] in (0, 1):  # end of text character or padding character
            continue
        at["id"] = next_token
        next_token += 1
        special_vocab[at["content"]] = at["id"]

    for special_token in [chr(288)]:
        special_vocab[special_token] = next_token
        next_token += 1

    for digit in range(10):
        special_vocab[str(digit)] = next_token
        next_token += 1

    for special_token in SPECIAL_TOKENS:
        actual_token = special_token
        if special_token not in (EOS_TOKEN, PADDING_TOKEN):
            actual_token = chr(288) + special_token

        letter_set = {x for x in actual_token}
        for letter in letter_set:
            if letter not in special_vocab:
                special_vocab[letter] = next_token
                next_token += 1

        for i in range(1, len(actual_token)):
            if actual_token[:i] not in special_vocab:
                special_vocab[actual_token[:i]] = next_token
                next_token += 1

    print("The real vocab starts on", next_token)
    return next_token, special_vocab


def dump_tokenizer(
    total_meaningful_tokens,
    starting_token=None,
    remappings_map=None,
    output_dir=None,
):
    '''
    Build the tokenizer json files for the specific dataset, and write them to disk.
    
    :param total_meaningful_tokens: The number of tokens we need for the dataset
    :param starting_token: Token number for the first non-grammar token.
    :param remappings_map: An auxiliary object that encodes the tokens reserved for
        subjects, relationships, objects, etc. for easy reading/debugging.
    :param output_dir: Where the data is written to.
    '''
    base_tokenizer_path = os.path.join(BASE_TOKENIZER_PATH, "tokenizer.json")

    with open(base_tokenizer_path, "r", encoding="utf-8") as f:
        base_tokenizer = json.load(f)
    config = copy.deepcopy(base_tokenizer)

    # Remap the IDs of special characters
    special_vocab = {}
    special_vocab[EOS_TOKEN] = 0
    special_vocab["<|padding|>"] = 1
    special_vocab["."] = 2
    next_token = 3
    for token in SPECIAL_TOKENS[2:]:
        special_vocab[chr(288) + token] = next_token
        next_token += 1
    for at in config["added_tokens"]:
        if at["id"] in (0, 1):  # end of text character or padding character
            continue
        at["id"] = next_token
        next_token += 1
        special_vocab[at["content"]] = at["id"]

    for special_token in [chr(288)]:
        special_vocab[special_token] = next_token
        next_token += 1

    for digit in range(10):
        special_vocab[str(digit)] = next_token
        next_token += 1

    merges = []

    for special_token in SPECIAL_TOKENS:
        actual_token = special_token
        if special_token not in (EOS_TOKEN, PADDING_TOKEN):
            actual_token = chr(288) + special_token

        letter_set = {x for x in actual_token}
        for letter in letter_set:
            if letter not in special_vocab:
                special_vocab[letter] = next_token
                next_token += 1
        for i in range(1, len(actual_token)):
            if actual_token[:i] not in special_vocab:
                special_vocab[actual_token[:i]] = next_token
                next_token += 1
            merges.append(f"{actual_token[:i]} {actual_token[i]}")

    if starting_token is not None and starting_token != next_token:
        raise ValueError(f"there was a mistake somewhere setting up IDs, {starting_token} {next_token}")

    merges_set = set()
    four_digit_tokens = []
    # The 4-digit tokens will start at next_token. The smaller-digit tokens will start at next_token + total_meaningful_tokens.
    for token in range(total_meaningful_tokens):
        updated = token + next_token
        four_digit_tokens.append(updated)
    next_token += total_meaningful_tokens
    for updated in four_digit_tokens:
        # Since the tokens are 4 digits long, we can only allow up to 10K tokens.
        if next_token > 9999:
            raise ValueError("only a dictionary of 10K tokens is supported")
        token_str = str(updated).zfill(4)
        prefix = chr(288) + token_str[0]
        if prefix not in merges_set:
            merges.append(f"{prefix[:-1]} {prefix[-1]}")
            merges_set.add(prefix)
            special_vocab[prefix] = next_token
            next_token += 1
        prefix = chr(288) + token_str[:2]
        if prefix not in merges_set:
            merges.append(f"{prefix[:-1]} {prefix[-1]}")
            merges_set.add(prefix)
            special_vocab[prefix] = next_token
            next_token += 1
        prefix = chr(288) + token_str[:3]
        if prefix not in merges_set:
            merges.append(f"{prefix[:-1]} {prefix[-1]}")
            merges_set.add(prefix)
            special_vocab[prefix] = next_token
            next_token += 1
        prefix = chr(288) + token_str
        if prefix not in merges_set:
            merges.append(f"{prefix[:-1]} {prefix[-1]}")
            merges_set.add(prefix)
    for updated in four_digit_tokens:
        token_str = chr(288) + str(updated).zfill(4)
        special_vocab[token_str] = updated

    config["model"]["merges"] = merges
    config["model"]["vocab"] = special_vocab

    if len(set(special_vocab.values())) < len(special_vocab):
        raise RuntimeError("there'sa  dduplicate key somewhere!")
    if max(special_vocab.values()) > len(special_vocab):
        raise RuntimeError(
            "we skipped a key somewhere!",
            set(range(len(special_vocab)))
            - set(special_vocab.values()),
        )

    with open(
        os.path.join(output_dir, "metadata", "tokenizer.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f)

    for f in ["tokenizer_config.json"]:
        shutil.copy(os.path.join(BASE_TOKENIZER_PATH, f), os.path.join(output_dir, "metadata", f))

    with open(
        os.path.join(output_dir, "metadata", "remappings_map.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(remappings_map, f)


def split_numerical_to_ids(num_needed, starting_value):
    '''
    Creates words for subject/relationship/object identifiers.
    If fewer than 1000 words are needed, they are one token long.
    If more than 1000 words are needed, they are two tokens long.
    
    :param num_needed: How many words are needed
    :param starting_value: Which value the tokens should start on
    '''
    if num_needed < 1000:
        return [
            [x + starting_value] for x in range(num_needed)
        ], num_needed + starting_value
    if num_needed > 1000000:
        raise ValueError("You want too many tokens!")
    # If the number needed is divisible by 1000, just do that. 
    if num_needed % 1000 == 0:
        divisor = 1000
    else:
        # Painfully look for the biggest divisor of num_needed by
        # starting with the square root and moving down.
        biggest_guess = math.floor(num_needed**0.5)
        divisor = None
        for i in range(biggest_guess):
            if num_needed % biggest_guess - i == 0:
                divisor = biggest_guess - i
        if num_needed // divisor > 1000:
            raise ValueError(f"Cannot evenly create two-token words for {num_needed} tokens.")
    first_tokens = set()
    for num in range(num_needed):
        first_token = num // divisor
        first_tokens.add(first_token)
    remaps = []
    for num in range(num_needed):
        remaps.append(
            [
                num // divisor + starting_value,
                num % divisor + starting_value + len(first_tokens),
            ]
        )
    return remaps, starting_value + len(first_tokens) + divisor


def dump_tokenized_graph(converted_edges, output_dir, prefix=""):
    '''
    Convert all graph edges to words, and write them to disk.
    
    :param converted_edges: Graph edges with the correct token IDs but as integers, not strings
    :param output_dir: The output directory
    :param prefix: If given, prepend this to the filename.
    '''
    with open(
        os.path.join(
            output_dir, "metadata", prefix + "relationship_graph_tokenized.txt"
        ),
        "w",
        encoding="utf-8",
    ) as f:
        for v in converted_edges:
            line = (
                ",".join([str(v).zfill(4) for v in v[0]])
                + "\t"
                + ",".join([str(v).zfill(4) for v in v[1]])
                + "\t"
                + ",".join([str(v).zfill(4) for v in v[2]])
                + "\t"
            )
            if len(v) >= 4:
                if v[3] is not None:
                    line += ",".join([str(v).zfill(4) for v in v[3]]) + "\t"
                else:
                    line += "\t"
            if len(v) >= 7:
                if v[4] is not None:
                    line += (
                        ",".join([str(v).zfill(4) for v in v[4]])
                        + "\t"
                        + ",".join([str(v).zfill(4) for v in v[5]])
                        + "\t"
                        + ",".join([str(v).zfill(4) for v in v[6]])
                        + "\t"
                    )
                else:
                    line += "\t\t\t"
            f.write(line + "\n")
    with open(
        os.path.join(
            output_dir, "metadata", prefix + "relationship_graph_tokenized.pkl"
        ),
        "wb",
    ) as f:
        pkl.dump(converted_edges, f, protocol=pkl.HIGHEST_PROTOCOL)


def tokenize_and_dump_graph(
    graph,
    subjects,
    relationships,
    objects,
    nested_objects=0,
    dump_graph=True,
    write_tokenizer=True,
    output_dir=None,
):
    '''
    Assign consecutive token IDs to all subjects, relationships, and objects,
    then dump the graph to disk.
    
    :param graph: The randomly generated graph edges.
    :param subjects: Number of subjects
    :param relationships: Number of relationships
    :param objects: Number of objects
    :param nested_objects: Number of nested objects, if any
    :param dump_graph: Whether to write the graph to disk
    :param write_tokenizer: Whether to write the tokenizer to disk
    :param output_dir: Where to write everything
    '''
    start_token, special_vocab = count_special_tokens()
    initial_start_token = copy.deepcopy(start_token)
    subj_obj_tokenized_edges = {}
    subj_nested_obj_tokenized_edges = {}
    obj_nested_obj_tokenized_edges = {}
    converted_edges = []
    all_remappings = {"special_vocab": special_vocab}
    subjects = subjects or len({e[0] for e in graph})
    # subject_remappings is an array of the same length as the number of subjects with the codes.
    subject_remappings, start_token = split_numerical_to_ids(subjects, start_token)
    all_remappings["subjects"] = subject_remappings
    relationships = relationships or len({e[1] for e in graph})
    relationship_remappings, start_token = split_numerical_to_ids(
        relationships, start_token
    )
    all_remappings["relationships"] = relationship_remappings
    object_remappings = []
    for _ in range(relationships):
        rel_object_remappings, start_token = split_numerical_to_ids(
            objects, start_token
        )
        object_remappings += rel_object_remappings
    all_remappings["objects"] = object_remappings
    if nested_objects > 0:
        # Subject-to-nested_object remappings
        # For each subject, the number of nested_object relationships will be the same
        # as the number of regular object relationships, because each relationship/object pair is mapped to a
        # nested_relationship/nested_object pair.
        nested_rel_remappings1, start_token = split_numerical_to_ids(
            relationships, start_token
        )
        all_remappings["subject_nested_relationships"] = nested_rel_remappings1
        # Object-to-nested_object remappings
        # The number of these is also the same as the number of relationships, because we use a different
        # one for each relationship (kind of an arbitrary choice)
        nested_rel_remappings2, start_token = split_numerical_to_ids(
            relationships, start_token
        )
        all_remappings["object_nested_relationships"] = nested_rel_remappings2
        nested_object_remappings = []
        for _ in range(relationships):
            rel_nested_object_remappings, start_token = split_numerical_to_ids(
                nested_objects, start_token
            )
            nested_object_remappings += rel_nested_object_remappings
        all_remappings["object_nested_objects"] = nested_object_remappings
    total_meaningful_tokens = start_token + 1  # start_token is 0-indexed

    # For simplicity, create all grammatical special tokens.
    total_meaningful_tokens, OTHER_SPECIAL_TOKENS = pc_lib.get_all_special_tokens(
        total_meaningful_tokens, {}
    )

    for i, edge in enumerate(graph):
        if i % 1000 == 0:
            print(i)
        subject_token = subject_remappings[edge[0]]
        relationship_token = relationship_remappings[edge[1]]
        object_token = object_remappings[edge[1] * objects + edge[2]]
        alt_object_token = None
        if len(edge) > 3 and edge[3] is not None:
            alt_object_token = object_remappings[edge[1] * objects + edge[3]]
        else:
            alt_object_token = None
        if nested_objects > 0:
            subj_nested_rel_token = nested_rel_remappings1[edge[1]]
            obj_nested_rel_token = nested_rel_remappings2[edge[1]]
            nested_obj_token = nested_object_remappings[
                edge[1] * nested_objects + edge[4]
            ]
            converted_edges.append(
                [
                    subject_token,
                    relationship_token,
                    object_token,
                    alt_object_token,
                    subj_nested_rel_token,
                    obj_nested_rel_token,
                    nested_obj_token,
                ]
            )
            subj_obj_tokenized_edges[f"{edge[1]}={edge[0]}"] = [
                [subject_token, relationship_token, object_token, None]
            ]
            subj_nested_obj_tokenized_edges[f"{edge[1]}={edge[0]}"] = [
                [subject_token, subj_nested_rel_token, nested_obj_token, None]
            ]
            obj_nested_obj_tokenized_edges[f"{edge[1]}={edge[2]}"] = [
                [object_token, obj_nested_rel_token, nested_obj_token, None]
            ]
        else:
            converted_edges.append(
                [subject_token, relationship_token, object_token, alt_object_token]
            )
            subj_obj_tokenized_edges[f"{edge[1]}={edge[0]}"] = [
                [subject_token, relationship_token, object_token, alt_object_token]
            ]

    if dump_graph:
        with open(
            os.path.join(output_dir, "metadata", "relationship_graph_tokenized.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for v in converted_edges:
                line = (
                    ",".join([str(v).zfill(4) for v in v[0]])
                    + "\t"
                    + ",".join([str(v).zfill(4) for v in v[1]])
                    + "\t"
                    + ",".join([str(v).zfill(4) for v in v[2]])
                    + "\t"
                )
                if len(v) >= 4:
                    if v[3] is not None:
                        line += ",".join([str(v).zfill(4) for v in v[3]]) + "\t"
                    else:
                        line += "\t"
                if len(v) >= 7:
                    if v[4] is not None:
                        line += (
                            ",".join([str(v).zfill(4) for v in v[4]])
                            + "\t"
                            + ",".join([str(v).zfill(4) for v in v[5]])
                            + "\t"
                            + ",".join([str(v).zfill(4) for v in v[6]])
                            + "\t"
                        )
                    else:
                        line += "\t\t\t"
                f.write(line + "\n")
        with open(
            os.path.join(output_dir, "metadata", "relationship_graph_tokenized.pkl"),
            "wb",
        ) as f:
            pkl.dump(converted_edges, f, protocol=pkl.HIGHEST_PROTOCOL)

    if write_tokenizer:
        dump_tokenizer(
            total_meaningful_tokens,
            initial_start_token,
            all_remappings,
            output_dir,
        )

    return (
        subj_obj_tokenized_edges,
        subj_nested_obj_tokenized_edges,
        obj_nested_obj_tokenized_edges,
        total_meaningful_tokens + 1,
        OTHER_SPECIAL_TOKENS,
        all_remappings,
    )


def dump_graph(graph, output_path):
    '''
    Write graph to disk as json
    '''
    with open(output_path + "json", "w", encoding="utf-8") as f:
        for item in graph:
            json_record = json.dumps(item)
            f.write(json_record + "\n")
    with open(output_path + ".pkl", "wb") as f:
        pkl.dump(graph, f, protocol=pkl.HIGHEST_PROTOCOL)
