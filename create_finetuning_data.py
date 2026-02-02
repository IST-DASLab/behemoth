import argparse
import copy
import datetime
import glob
import json
import math
import os
import random

import numpy as np

import graph_utils
import phrase_creation_utils as pc_lib


def load_subject_relationship_object_remappings(graph_dir):
    '''
    Load the dictionary of subject, relationship, and object remappings into memory.
    
    :param graph_dir: Path to main data dir.
    '''
    with open(os.path.join(graph_dir, 'metadata', 'remappings_map.json'), 'r') as f:
        remappings = json.load(f)
    return remappings
    
def tokenize_edge(edge, remappings, edge_type = "so"):
    '''
    Converts a graph edge to its token representation
    
    :param edge: The edge to tokenize
    :param remappings: Dictionary of ordinal-to-token remappings
    :param edge_type: Type of edge: so for subject-object, sno for subject-nested object, ono for object-nested object
    '''

    if edge_type == 'so':
        num_objects = len(remappings['objects']) // len(remappings['relationships'])
        subject_token = remappings['subjects'][edge[0]]
        relationship_token = remappings['relationships'][edge[1]]
        object_token = remappings['objects'][edge[1] * num_objects + edge[2]]
        return [subject_token, relationship_token, object_token]
    elif edge_type == 'sno':
        if 'subject_nested_relationships' not in remappings:
            raise ValueError("Subject-Nested Object edge requested but not possible")
        subject_token = remappings['subjects'][edge[0]]
        num_nested_objects = len(remappings['object_nested_objects']) // len(remappings['relationships'])
        subj_nested_rel_token = remappings['subject_nested_relationships'][edge[1]]
        nested_obj_token = remappings['object_nested_objects'][
            edge[1] * num_nested_objects + edge[2]
        ]
        return [subject_token, subj_nested_rel_token, nested_obj_token]
    elif edge_type == 'ono':
        if 'object_nested_relationships' not in remappings:
            raise ValueError("Object-Nested Object edge requested but not possible")
        num_objects = len(remappings['objects']) // len(remappings['relationships'])
        object_token = remappings['objects'][edge[1] * num_objects + edge[0]]
        num_nested_objects = len(remappings['object_nested_objects']) // len(remappings['relationships'])
        obj_nested_rel_token = remappings['object_nested_relationships'][edge[1]]
        nested_obj_token = remappings['object_nested_objects'][
            edge[1] * num_nested_objects + edge[2]
        ]
        return [object_token, obj_nested_rel_token, nested_obj_token]
    raise ValueError("Only supported values for edge type are so, sno, or ono")


def write_validation_data(
    tokenized_edges,
    phrase_creator,
    qa_finetuning_phrase_creator=None,
    subjects_with_questions=[],
    output_suffix="",
    file_prefix="",
):
    '''
    We produce the following validation data:
    A list of sentences in standard format (using phrase_creator), with template ID, if applicable. In cases with disagreements, the modal label is correct.
    TODO: Ditto, for cases of disagreement
    A list of question phrases, for cases where the subject was in the question phrases list
    A list of question phrases, for cases where the subject was *not* in question phrases list
    
    :param tokenized_edges: Edges to write as sentences
    :param phrase_creator: The phrase creator used to create the sentences
    :param qa_finetuning_phrase_creator: Ditto, for QA format
    :param subjects_with_questions: Which subjects should have their sentences be in QA format
    :param output_suffix: A string to prepend to the output directory
    :param file_prefix: A string to prepend to the file name.

    '''
    outputs = {}
    for token_matchings in tokenized_edges.values():
        for subject_token, relationship_token, object_token in token_matchings:
            val_phrases = phrase_creator.create_val_phrase(
                subject_token, relationship_token, object_token
            )
            # There might be separate types of templates
            for phrase in val_phrases:
                phrase_type = phrase[0]
                if phrase_type in outputs:
                    outputs[phrase_type].append(phrase[1:3])
                else:
                    outputs[phrase_type] = [phrase[1:3]]
    if args.ft_subdir is not None:
        output_suffix = os.path.join(args.ft_subdir, output_suffix)
    for k, v in outputs.items():
        with open(
            os.path.join(
                args.output_dir, f"{output_suffix}validations", f"{file_prefix}{k}.txt"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(["\t".join(x) for x in v]) + "\n")
    questions = {}
    new_questions = {}
    if original_args["questions_frac"] > 0 and qa_finetuning_phrase_creator is not None:
        for o_token, token_matchings in tokenized_edges.items():
            for subject_token, relationship_token, object_token, _ in token_matchings:
                question_phrases = qa_finetuning_phrase_creator.create_val_phrase(
                    subject_token, relationship_token, object_token
                )
                for phrase in question_phrases:
                    phrase_type = phrase[0]
                    if o_token in subjects_with_questions:
                        if phrase_type in questions:
                            questions[phrase_type].append(phrase[1:3])
                        else:
                            questions[phrase_type] = [phrase[1:3]]
                    else:
                        if phrase_type in new_questions:
                            new_questions[phrase_type].append(phrase[1:3])
                        else:
                            new_questions[phrase_type] = [phrase[1:3]]

        output_dir = args.output_dir
        for k, v in questions.items():
            with open(
                os.path.join(
                    output_dir, f"{output_suffix}questions", f"{file_prefix}{k}.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("\n".join(["\t".join(x) for x in v]) + "\n")
        for k, v in new_questions.items():
            with open(
                os.path.join(
                    output_dir, f"{output_suffix}questions", f"{file_prefix}{k}_new.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("\n".join(["\t".join(x) for x in v]) + "\n")


def write_ft_data(tokenized_edges, phrase_creator, prefix="", ft_subdir=None):
    '''
    Write finetuning data to a directory.
    
    :param tokenized_edges: Edges to write as sentences
    :param phrase_creator: The class that turns edges into sentences
    :param prefix: Prefix to prepend to the data file names
    :param ft_subdir: Name of subdirectory to write files to
    '''
    val_outputs = {}
    phrases = []
    # Also write the outputs in graph form.
    graph_utils.dump_tokenized_graph(
        [k[0] for k in tokenized_edges.values()],
        os.path.join(args.output_dir, ft_subdir),
        prefix,
    )
    for token_matchings in tokenized_edges.values():
        for subject_token, relationship_token, object_token in token_matchings:
            phrases.append(
                phrase_creator.create_phrase(
                    subject_token, relationship_token, object_token
                )
            )

            # The val phrase creator is the correct format for finetuning.
            val_phrases = phrase_creator.create_val_phrase(
                subject_token, relationship_token, object_token
            )
            # There might be separate types of templates
            for phrase in val_phrases:
                phrase_type = phrase[0]
                if phrase_type in val_outputs:
                    val_outputs[phrase_type].append(phrase[1:3])
                else:
                    val_outputs[phrase_type] = [phrase[1:3]]
    with open(
        os.path.join(
            args.output_dir, ft_subdir, "finetuning_data", f"{prefix}data.txt"
        ),
        "w",
        encoding="utf-8",
    ) as f:
        print(
            "writing FT data to",
            os.path.join(
                args.output_dir, ft_subdir, "finetuning_data", f"{prefix}data.txt"
            ),
        )
        f.write("\n".join([x for x in phrases]) + "\n")
    for k, v in val_outputs.items():
        with open(
            os.path.join(
                args.output_dir, ft_subdir, "ft_validations", f"{prefix}{k}.txt"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(["\t".join(x) for x in v]) + "\n")
    for k, v in val_outputs.items():
        with open(
            os.path.join(
                args.output_dir, ft_subdir, "finetuning_data", f"{prefix}{k}.jsonl"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            lines = []
            for q, a in v:
                line = json.dumps({"instruction": q, "input": "", "output": a})
                lines.append(line)
            f.write("\n".join(lines))
            f.write("\n")


def get_phrase_creators(total_meaningful_tokens, special_tokens):
    '''
    Get the necessary phrase creators to create finetuning and validation data.
    
    :param total_meaningful_tokens: Number of tokens used to create this particular dataset and phrases.
    :param special_tokens: Already known tokens (phrase creator will create any additional ones.)
    '''
    if original_args["questions_entity_frac"] or args.add_qa_data_to_finetuning:
        qa_finetuning_phrase_creator = pc_lib.SimpleQuestionPhraseCreator(
            total_meaningful_tokens, special_tokens
        )
    else:
        qa_finetuning_phrase_creator = None
    if (
        args.finetuning_phrase_creator is not None
        and args.finetuning_phrase_creator != ""
    ):
        # To teach a new QA format
        if args.finetuning_phrase_creator == "qa":
            finetuning_phrase_creator = pc_lib.SimpleFinetuningPhraseCreator(
                total_meaningful_tokens, special_tokens
            )
        if args.finetuning_phrase_creator == "qa_refusal":
            finetuning_phrase_creator = pc_lib.SimpleFinetuningQARefusalPhraseCreator(
                total_meaningful_tokens, special_tokens
            )
        if args.finetuning_phrase_creator == "refusal":
            finetuning_phrase_creator = pc_lib.SimpleFinetuningRefusalPhraseCreator(
                total_meaningful_tokens, special_tokens
            )
        if args.finetuning_phrase_creator == "remapping":
            finetuning_phrase_creator = pc_lib.SimpleInvertedPhraseCreator(
                total_meaningful_tokens, special_tokens, inversion_frac=0
            )
    return finetuning_phrase_creator, qa_finetuning_phrase_creator


def pick_edges_with_desired_relationship(edges, relationship_ordinal, fraction):
    '''
    Find all edges in the graph that have a specific relationship value.
    
    :param edges: List of all edges.
    :param relationship_ordinal: Which relationship we want.
    :param fraction: If specified, only return that fraction of possible edges.

    '''
    relevant_edges = [
        x for x in edges if x[1] == relationship_ordinal
    ]
    if fraction < 1.0:
        random.shuffle(relevant_edges)
        relevant_edges = relevant_edges[
            : math.ceil(len(relevant_edges) * args.single_ft_data_subset_frac)
        ]
    return relevant_edges


def override_object_token_to_smallest(edges):
    '''
    For a set of edges (e.g., all those with a specific relationship, change the object to the smallest value.)
    
    :param edges: all edges for whom we want to override object. 

    '''
    relevant_edges = [[x[0], x[1], 0, None] for x in edges]
    return relevant_edges


def get_deduped_list_of_arrays(arrays):
    '''
    Removes duplicates from a list of arrays. Returns the elements sorted.
    
    :param arrays: The list of arrays to be deduped.
    '''
    arrays.sort()
    deduped_list = [arrays[0]]
    for ot in arrays[1:]:
        for i, t in enumerate(ot):
            if deduped_list[-1][i] != t:
                deduped_list.append(ot)
                continue
    return deduped_list


def get_object_tokens(edges, relationship_token):
    '''
    Returns all object tokenizations for a specific relationship.
    
    :param edges: Graph edges
    :param relationship_token: Token corresponding to relationship

    '''
    object_tokens = [x[2] for x in edges if x[1] == relationship_token]
    return get_deduped_list_of_arrays(object_tokens)


def get_edges_with_relationship_and_object(edges, relationship, num_objects, object=None):
    '''
    Get all edges that have a specific relationship and object.
    
    :param edges: The graph edges.
    :param relationship: The relationship in token form.
    :param num_objects: Total number of objects per relationship.
    :param object: The object in token form.
    '''
    if object is None:
        object = np.random.randint(0, num_objects)
    relevant_edges = [
        x for x in edges if x[1] == relationship and x[2] == object
    ]
    return relevant_edges, object


def override_object_token(edges, object_token):
    '''
    Change the object token in a set of edges to a new value.
    
    :param edges: Set of graph edges
    :param object_token: New object token
    '''
    return [[x[0], x[1], object_token, None] for x in edges]


def filter_for_correlated_tuples(
    relevant_edges, all_edges
):
    '''
    Filter a set of edges to those for which the second object matches the first object. 
    
    :param relevant_edges: Edges which should be filtered
    :param all_edges: All edges
    '''
    # Find all edges with the second relationship.
    edges_with_second_relationship = {e[0]: e[2] for e in pick_edges_with_desired_relationship(all_edges, 1, 1.)}
    correlated_edges = [r for r in relevant_edges if r[2] == edges_with_second_relationship[r[0]]]
    #uncorrelated_edges_first_rel = [r for r in relevant_edges if r[2] != edges_with_second_relationship[r[0]]]
    #correlated_edge_subjects = {r[0] for r in correlated_edges}
    #correlated_edges_first_rel = [[k, 0, v for k, v in edges_with_second_relationship.items() if k not in correlated_edge_subjects]
    #uncorrelated_edges_second_rel = [[k, 1, v] for k, v in edges_with_second_relationship.items() if k not in correlated_edge_subjects]
        
    return relevant_edges


def filter_for_different_object_values(relevant_edges):
    '''
    For a set of edges, keep only one with each object value.
    
    :param relevant_edges: Iterable of graph edges.
    '''
    seen_objects = set()
    filtered_edges = []
    for relevant_edge in relevant_edges:
        if str(relevant_edge[2]) in seen_objects:
            continue
        seen_objects.add(str(relevant_edge[2]))
        filtered_edges.append(relevant_edge)
    return filtered_edges


def partition_edges_into_converted_and_clean(edges, all_different=False):
    if all_different:
        filtered_set = filter_for_different_object_values(edges)
    else:
        filtered_set = edges

    selected_edges = random.sample(filtered_set, args.num_ft_overrides)
    selected_edge_subjects = {str(x[0]) for x in selected_edges}
    other_edges = [
        copy.deepcopy(e) for e in edges if str(e[0]) not in selected_edge_subjects
    ]

    random.shuffle(other_edges)
    return selected_edges, other_edges


def remove_edges_with_affected_subjects(edges, converted_edges):
    converted_edge_subjects = {str(x[0]) for x in converted_edges}
    other_edges = [
        copy.deepcopy(e) for e in edges if str(e[0]) not in converted_edge_subjects
    ]

    random.shuffle(other_edges)
    return other_edges


def get_nested_obj_edges_matching_sample(
    post_conversion_edges,
    pre_conversion_edges,
    subj_obj_edges,
    subj_nested_obj_edges,
    obj_nested_obj_edges,
    remappings
):
    test_edges = {}
    same_subj_obj_edges = []
    same_subj_nested_obj_edges = []
    same_obj_nested_obj_edges = []
    subj_remapped_nested_obj_edges = []
    remapped_obj_nested_obj_edges = []
    if args.remap_subject_nested_object_edges:
        if args.single_ft_value:
            raise NotImplementedError("Remapping the nested object edges is not implemented for remapping to single value.")
        for edge_to_convert in pre_conversion_edges:
            for soe in subj_obj_edges:
                if soe[0] == edge_to_convert[0] and soe[1] == args.relationship_ordinal:
                    same_subj_obj_edges.append(soe)
                    break
            for ome in obj_nested_obj_edges:
                if ome[0] == edge_to_convert[2] and ome[1] == args.relationship_ordinal:
                    same_obj_nested_obj_edges.append(ome)
                    break
        for edge_to_convert in post_conversion_edges:
            for ome in obj_nested_obj_edges:
                if ome[0] == edge_to_convert[2] and ome[1] == args.relationship_ordinal:
                    remapped_obj_nested_obj_edges.append([edge_to_convert[0], edge_to_convert[1], ome[2]])
                    break
    elif args.remap_object_nested_object_edges:
        if args.single_ft_value:
            raise ValueError("not implemented")
        same_subj_obj_edges = []
        same_subj_nested_obj_edges = []
        for edge_to_convert in pre_conversion_edges:
            for soe in subj_obj_edges:
                if soe[2] == edge_to_convert[0] and soe[1] == args.relationship_ordinal:
                    same_subj_obj_edges.append(soe)
                    break
            for sme in subj_nested_obj_edges:
                if sme[2] == edge_to_convert[2] and sme[1] == args.relationship_ordinal:
                    same_subj_nested_obj_edges.append(sme)
                    break
        for edge_to_convert in post_conversion_edges:
            for sme in subj_nested_obj_edges:
                if sme[2] == edge_to_convert[0] and sme[1] == args.relationship_ordinal:
                    subj_remapped_nested_obj_edges.append([[sme[0], sme[1], edge_to_convert[2]]])
                    break
    else: # Remapping subject-object edge.
        for edge_to_convert in pre_conversion_edges:
            for sme in subj_nested_obj_edges:
                if sme[0] == edge_to_convert[0] and sme[1] == args.relationship_ordinal:
                    same_subj_nested_obj_edges.append(sme)
                    break
            for ome in obj_nested_obj_edges:
                if ome[0] == edge_to_convert[2] and ome[1] == args.relationship_ordinal:
                    same_obj_nested_obj_edges.append(ome)
                    break
        for edge_to_convert in post_conversion_edges:
            for ome in obj_nested_obj_edges:
                if ome[0] == edge_to_convert[2] and ome[1] == args.relationship_ordinal:
                    remapped_obj_nested_obj_edges.append(ome)
                    break
    if len(same_subj_obj_edges) > 0:
        test_edges["same_subj_obj_edges"] = [tokenize_edge(e, remappings, 'so') for e in same_subj_obj_edges]
    if len(same_subj_nested_obj_edges) > 0:
        test_edges["same_subj_nested_obj_edges"] = [tokenize_edge(e, remappings, 'sno') for e in same_subj_nested_obj_edges]
    if len(same_obj_nested_obj_edges) > 0:
        test_edges["same_obj_nested_obj_edges"] = [tokenize_edge(e, remappings, 'ono') for e in same_obj_nested_obj_edges]
    if len(subj_remapped_nested_obj_edges) > 0:
        test_edges["subj_remapped_nested_obj_edges"] = [tokenize_edge(e, remappings, 'sno') for e in subj_remapped_nested_obj_edges]
    if len(remapped_obj_nested_obj_edges) > 0:
        test_edges["remapped_obj_nested_obj_edges"] = [tokenize_edge(e, remappings, 'ono') for e in remapped_obj_nested_obj_edges]
    return test_edges


# Assign consecutive token IDs to all subjects, relationships, and objects
# The object IDs are unique per relationship
def create_data(
    graph_edges,
    remappings,
    relationship_ordinal,
    ft_subdir=None,
    special_tokens=None,
    total_meaningful_tokens=0,
):
    if special_tokens is None:
        raise ValueError("Special tokens cannot be none")



    num_subjects = len(remappings["subjects"])
    num_relationships = len(remappings["relationships"])
    num_objects = len(remappings["objects"]) // len(remappings["relationships"])
    if 'subjects_nested_relationships' in remappings:
        num_nested_objects = len(remappings['object_nested_objects']) // len(remappings['object_nested_relationships'])

    num_override_rows = args.num_repeats_per_override * args.num_ft_overrides
    if num_override_rows > num_subjects:
        raise ValueError("Too many override rows to balance the finetuning data!")

    if relationship_ordinal >= num_relationships:
        raise ValueError("The requested relationship doesn't exist.")


    # Now we actually create the data.

    have_nested_objects = graph_edges[0][4] is not None

    if have_nested_objects:
        # Split up graph into normal edges, subject nested_object edges, and relationship nested_object edges.
        subj_obj_edges = [ce[:4] for ce in graph_edges]
        subj_nested_obj_edges = [[ce[0], ce[1], ce[4], None] for ce in graph_edges]
        obj_nested_obj_edges = [[ce[2], ce[1], ce[4], None] for ce in graph_edges]
        obj_nested_obj_edges = get_deduped_list_of_arrays(obj_nested_obj_edges)
    else:
        subj_obj_edges = copy.deepcopy(graph_edges)
        subj_nested_obj_edges = []
        obj_nested_obj_edges = []

    if (
        args.remap_subject_nested_object_edges or args.remap_object_nested_object_edges
    ) and not have_nested_objects:
        raise ValueError(
            "requesting nested_object remappings, but this dataset doesn't have any."
        )
    
    edge_type = 'so'
    changing_nested_objects = False
    if args.remap_subject_nested_object_edges:
        edges_for_remapping = subj_nested_obj_edges
        edge_type = 'sno'
        changing_nested_objects = True
    elif args.remap_object_nested_object_edges:
        edges_for_remapping = obj_nested_obj_edges
        edge_type = 'ono'
        changing_nested_objects = True
    else:
        edges_for_remapping = subj_obj_edges

    # This overwrites all edges to have the same value
    if args.single_ft_value:
        relevant_edges = (
            pick_edges_with_desired_relationship(
                edges_for_remapping, relationship_ordinal,
                args.single_ft_data_subset_frac
            )
        )
        relevant_edges_pre_conversion = copy.deepcopy(relevant_edges)
        relevant_edges = override_object_token_to_smallest(
            relevant_edges
        )
        num_override_rows = len(relevant_edges)

        train_edge_subjects = {e[0] for e in relevant_edges} 
        train_edges = {"censored": [tokenize_edge(e, remappings, edge_type) for e in relevant_edges]}

        # Store unrelated test edges
        # Remove the edges for the subject, even if the relationship is different.
        clean_subjects = list({
            x[0] for x in edges_for_remapping if x[0] not in train_edge_subjects
        })
        random.shuffle(clean_subjects)
        clean_subjects_for_train = set(clean_subjects[:len(clean_subjects)//2])
        clean_subjects_for_test = set(clean_subjects[len(clean_subjects)//2:])

        clean_edges_for_train = [e for e in edges_for_remapping if e[0] in clean_subjects_for_train and e[1] != relationship_ordinal]
        random.shuffle(clean_edges_for_train)
        train_edges["reg_data"] = [tokenize_edge(e, remappings, edge_type) for e in clean_edges_for_train[:len(relevant_edges)]]
        
        test_edges = {}
        clean_edges_for_test = [e for e in edges_for_remapping if e[0] in clean_subjects_for_test and e[1] != relationship_ordinal]
        random.shuffle(clean_edges_for_test)
        implicit_remaps = [e for e in edges_for_remapping if e[0] in clean_subjects_for_test and e[1] == relationship_ordinal]

        test_edges["reg_data"] = clean_edges_for_test[:10000]
        test_edges["reg_data"] = [tokenize_edge(e, remappings, edge_type) for e in clean_edges_for_test[:10000]]
        test_edges["implicitly_remapped_edges"] = [tokenize_edge(e, remappings, edge_type) for e in implicit_remaps[:10000]]


        if len(subj_nested_obj_edges) > 0 and args.remap_subject_nested_object_edges:
            nested_test_edges = get_nested_obj_edges_matching_sample(
                relevant_edges,
                relevant_edges_pre_conversion,
                subj_obj_edges,
                subj_nested_obj_edges,
                obj_nested_obj_edges,
                remappings
            )
            test_edges.update(nested_test_edges)

    elif args.num_ft_overrides > 0:

        relevant_edges = (
            pick_edges_with_desired_relationship(
                edges_for_remapping, args.relationship_ordinal, fraction=1.
            )
        )
        
        # If required, get the edges where the first and second relationship matches.
        if args.get_correlated_tuples:
                relevant_edges = filter_for_correlated_tuples(
                relevant_edges,
                edges_for_remapping,
            )
        

        # Now actually sample from the relevant edges and do the conversion.
        # If we don't want all overrides to be the same, then we want all initial objects to be different.
        # Of course, if the first object is binarized, we can't do that.
        filter_for_unique_objects = \
            ( not original_args["binarize_first_correlated"] ) and \
            (not args.all_same_remapping)

        if args.all_same_remapping:
            edges_with_rel_and_object, _ = get_edges_with_relationship_and_object(
                relevant_edges, relationship_ordinal, num_objects)
            edges_to_convert, other_edges_with_same_rel_and_object = (
                partition_edges_into_converted_and_clean(
                    edges_with_rel_and_object, all_different=False
                )
            )
            other_edges_same_rel = remove_edges_with_affected_subjects(relevant_edges, edges_to_convert)
            other_edges = remove_edges_with_affected_subjects(edges_for_remapping, edges_to_convert)
        else:
            edges_to_convert, other_edges_same_rel = partition_edges_into_converted_and_clean(
                relevant_edges, all_different=filter_for_unique_objects
            )
            other_edges = remove_edges_with_affected_subjects(edges_for_remapping, edges_to_convert)

        random.shuffle(other_edges_same_rel)
        random.shuffle(other_edges)

        edges_to_convert_pre_conversion = copy.deepcopy(edges_to_convert)

        train_edge_subjects = set()
        train_edges = {
            "edited": [],
            "with_same_value": [],
        }
        test_edges = {
            "edited": [],
            "with_same_value": [],
            "same_edges_diff_rel": [],
            "same_subject_second_rel": [],
            "same_subject_second_rel_converted": [],
        }

        # Change the object(s).
        if args.all_same_remapping:
            train_edge_subjects = train_edge_subjects.union({e[0] for e in edges_to_convert})
            used_object_tokens = {r[2] for r in edges_to_convert}
            if changing_nested_objects:
                new_object = random.choice([x for x in range(num_nested_objects) if x not in used_object_tokens])
            else:
                new_object = random.choice([x for x in range(num_objects) if x not in used_object_tokens])

            num_edges = len(other_edges_with_same_rel_and_object)
            lim = np.min([num_edges // 2, 150])
            train_edge_subjects.update({e[0] for e in other_edges_with_same_rel_and_object[:lim]})
            multiplier = args.num_ft_overrides
            train_edges["with_same_value"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_with_same_rel_and_object[:lim]] * multiplier
            test_edges["with_same_value"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_with_same_rel_and_object[lim : 2 * lim]]

            num_edges = len(other_edges_same_rel)
            lim = np.min([num_edges // 2, 500 * args.num_ft_overrides])
            train_edge_subjects.update({e[0] for e in other_edges_same_rel[:lim]})
            multiplier = 250 * args.num_ft_overrides // lim
            train_edges["with_same_rel"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_rel[:lim]]
            test_edges["with_same_rel"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_rel[lim : 2 * lim]]
            
            other_edges = [e for e in edges_for_remapping if e[0] not in train_edge_subjects]
            random.shuffle(other_edges)
            other_edges_train = [e for e in other_edges[:len(other_edges)//2] if e[1] != relationship_ordinal]
            other_edges_test = other_edges[len(other_edges)//2:len(other_edges) + 10000]
            train_edges["reg_data"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_train[:num_override_rows]]
            test_edges["reg_data"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_test]

            edges_to_convert = [[x[0], x[1], new_object, None] for x in edges_to_convert]
            for edge_to_convert in edges_to_convert:
                converted = [edge_to_convert[0], edge_to_convert[1], new_object]
                train_edges["edited"] += [
                    tokenize_edge(converted, remappings, edge_type)
                ] * args.num_repeats_per_override
                test_edges["edited"] += [
                    tokenize_edge(converted, remappings, edge_type)
                ]
                #raise ValueError(train_edges.keys())

                test_edges["same_edges_diff_rel"] += [
                    tokenize_edge(x, remappings, edge_type)
                    for x in edges_for_remapping
                    if x[0] == edge_to_convert[0] and x[1] != relationship_ordinal
                ]
                if args.get_correlated_tuples:
                    test_edges["same_subject_second_rel"] += [
                        tokenize_edge(x, remappings, edge_type)
                        for x in edges_for_remapping
                        if x[0] == edge_to_convert[0]
                        and x[1] == 1
                    ]
                    test_edges["same_subject_second_rel_converted"] += [
                        tokenize_edge([x[0], x[1], new_object], remappings, edge_type)
                        for x in graph_edges
                        if x[0] == edge_to_convert[0]
                        and x[1] == 1
                    ]

        # If different remappings, make different ones for all the conversions.
        elif (not original_args["binarize_first_correlated"]) or args.num_ft_overrides <= 2:
            used_object_tokens = {r[2] for r in edges_to_convert}
            if changing_nested_objects:
                new_objects = np.random.choice([x for x in range(num_nested_objects) if x not in used_object_tokens], args.num_ft_overrides, replace=False)
            else:
                new_objects = np.random.choice([x for x in range(num_objects) if x not in used_object_tokens], args.num_ft_overrides, replace=False)
            random.shuffle(new_objects)

            edges_to_convert = [[x[0], x[1], new_objects[i], None] for i, x in enumerate(edges_to_convert)]
            train_edge_subjects = train_edge_subjects.union({e[0] for e in edges_to_convert})
            for i, edge_to_convert in enumerate(edges_to_convert):
                converted = [edge_to_convert[0], edge_to_convert[1], new_objects[i]]
                train_edges["edited"] += [
                    tokenize_edge(converted, remappings, edge_type)
                ] * args.num_repeats_per_override
                test_edges["edited"] += [
                    tokenize_edge(converted, remappings, edge_type)
                ]

                test_edges["same_edges_diff_rel"] += [
                    tokenize_edge(x, remappings, edge_type)
                    for x in edges_for_remapping
                    if x[0] == edge_to_convert[0] and x[1] != relationship_ordinal
                ]

                other_edges_same_value = [
                    e for e in edges_for_remapping if e[1] == relationship_ordinal and e[2] == edge_to_convert[2]
                ]
                random.shuffle(other_edges_same_value)
                num_edges = len(other_edges_same_value)
                lim = np.min([num_edges // 2, 150])
                train_edge_subjects.update({e[0] for e in other_edges_same_value[:lim]})
                train_edges["with_same_value"] += [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_value[:lim]]
                test_edges["with_same_value"] += [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_value[lim: 2 * lim]]

                num_edges = len(other_edges_same_rel)
                lim = np.min([num_edges // 2, 500 * args.num_ft_overrides])
                multiplier = 250 * args.num_ft_overrides // lim
                train_edge_subjects.update({e[0] for e in other_edges_same_rel[:lim]})
                train_edges["with_same_rel"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_rel[:lim]]
                test_edges["with_same_rel"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_rel[lim : 2 * lim]]
                
                other_edges = [e for e in edges_for_remapping if e[0] not in train_edge_subjects]
                random.shuffle(other_edges)
                other_edges_train = [e for e in other_edges[:len(other_edges)//2] if e[1] != relationship_ordinal]
                other_edges_test = other_edges[len(other_edges)//2:len(other_edges) + 10000]
                train_edges["reg_data"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_train[:num_override_rows]]
                test_edges["reg_data"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_test]

                if args.get_correlated_tuples:
                    test_edges["same_subject_second_rel"] += [
                        tokenize_edge(x, remappings, edge_type)
                        for x in edges_for_remapping
                        if x[0] == edge_to_convert[0]
                        and x[1] == 1
                    ]
                    test_edges["same_subject_second_rel_converted"] += [
                        tokenize_edge([x[0], x[1], new_objects[i]], remappings, edge_type)
                        for x in edges_for_remapping
                        if x[0] == edge_to_convert[0]
                        and x[1] == 1
                    ]
        else:
            # Handle the binarized case.
            counts = {k: 0 for k in [0, 1]}
            for edge_to_convert in edges_to_convert:
                counts[edge_to_convert[2]] += 1
                edge_to_convert[2] = 1-edge_to_convert[2]
            for k, count in counts.items():
                other_edges_same_value = [e for e in other_edges if e[2] == k]
                random.shuffle(other_edges_same_value)
                lim = np.min([len(other_edges_same_value) // 2, 150 * count])
                train_edges["with_same_value"] += [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_value[:lim]]
                test_edges["with_same_value"] += [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_value[lim: 2 * lim]]
                
                num_edges = len(other_edges_same_rel)
                lim = np.min([num_edges // 2, 500 * args.num_ft_overrides])
                multiplier = 250 * args.num_ft_overrides // lim
                train_edges["with_same_rel"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_rel[:lim]]
                test_edges["with_same_rel"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_same_rel[lim : 2 * lim]]
                
                all_affected_subjects = {e[0] for e in edges_to_convert}
                for edges in train_edges.values():
                    all_affected_subjects = all_affected_subjects.union(e[0] for e in edges)
                other_edges = [e for e in edges_for_remapping if e not in all_affected_subjects]
                random.shuffle(other_edges)
                other_edges_train = [e for e in other_edges[:len(other_edges)//2] if e[1] != relationship_ordinal]
                other_edges_test = other_edges[len(other_edges)//2:len(other_edges) + 10000]
                train_edges["reg_data"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_train[:num_override_rows]]
                test_edges["reg_data"] = [tokenize_edge(e, remappings, edge_type) for e in other_edges_test]

        # # Create some training data.
        # num_edges = len(other_edges)
        
        # lim = np.min([num_edges // 2, 500 * args.num_ft_overrides])
        # multiplier = 250 * args.num_ft_overrides // lim
        # train_edges[""] = [tokenize_edge(x, remappings, edge_type) for x in edges_to_convert] * args.num_repeats_per_override
        # train_edges["other_edges"] = [tokenize_edge(x, remappings, edge_type) for x in other_edges[:lim]]

        # test_edges[""] = [tokenize_edge(x, remappings, edge_type) for x in edges_to_convert] * args.num_repeats_per_override
        # test_edges["other_edges"] = [tokenize_edge(x, remappings, edge_type) for x in other_edges[lim : 2 * lim]]

        if len(subj_nested_obj_edges) > 0:
            if relationship_ordinal > 0:
                raise ValueError("Relationship picker not implemented for this case.")
            nested_test_edges = get_nested_obj_edges_matching_sample(
                edges_to_convert,
                edges_to_convert_pre_conversion,
                subj_obj_edges,
                subj_nested_obj_edges,
                obj_nested_obj_edges,
                remappings
            )
            test_edges.update(nested_test_edges)

        
        # if len(same_edges_second_rel) > 0:
        #     rels = list(set(str(x[1]) for x in same_edges_second_rel))
        #     for rel in rels:
        #         test_edges[f"same_subj_second_rel"] = [
        #             x for x in same_edges_second_rel if str(x[1]) == rel
        #         ]
        # affected_subjects = {str(x[0]) for x in edges_to_convert}
        # clean_edges = [
        #     x for x in edges_for_remapping if str(x[0]) not in affected_subjects
        # ]
        # rels = np.unique([x[1] for x in graph])
        # test_edges.update({r: [x for x in clean_edges if x[1] == r] for r in rels})

    # This is a hack, sadly
    train_edges = {
        k: {str(i): [x] for i, x in enumerate(v)} for k, v in train_edges.items()
    }
    test_edges = {
        k: {str(i): [x] for i, x in enumerate(v)} for k, v in test_edges.items()
    }

    finetuning_phrase_creator, qa_finetuning_phrase_creator = get_phrase_creators(
        total_meaningful_tokens, special_tokens
    )

    for k, v in train_edges.items():
        print("writing file", k, len(v), ft_subdir)
        write_ft_data(
            v,
            finetuning_phrase_creator,
            ft_subdir=ft_subdir,
            prefix=f"{k}_" if k else "",
        )
        if args.add_qa_data_to_finetuning:
            write_ft_data(
                v, qa_finetuning_phrase_creator, prefix="{k}_qa_", ft_subdir=ft_subdir
            )
    all_lines = []
    for training_file in glob.glob(
        os.path.join(args.output_dir, ft_subdir, "finetuning_data", "*.jsonl")
    ):
        # Handle the case of overwriting the same directory.
        if os.path.basename(training_file) in (
            "train.jsonl",
            "test.jsonl",
            "val.jsonl",
        ):
            continue
        with open(training_file, "r", encoding="utf-8") as f:
            all_lines += f.readlines()
    for type in ["train", "test", "val"]:
        with open(
            os.path.join(
                args.output_dir, ft_subdir, "finetuning_data", f"{type}.jsonl"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines(all_lines)

    for k, v in test_edges.items():
        print("writing file", k, len(v), ft_subdir)
        write_validation_data(
            v,
            finetuning_phrase_creator,
            qa_finetuning_phrase_creator,
            output_suffix=f"ft_",
            file_prefix=f"{k}_",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data generator arguments")
    parser.add_argument(
        "-g",
        "--graph-path",
        type=str,
        default=None,
        help="Path to load graph from, rather than generating random.",
    )
    parser.add_argument(
        "-f",
        "--finetuning-phrase-creator",
        type=str,
        default="remapping",
        help="if activated, creates a finetuning dataset.",
    )
    parser.add_argument(
        "--single-ft-data-subset-frac",
        type=float,
        default=1.0,
        help="Subset the data with single finetune value.",
    )
    parser.add_argument(
        "--add-qa-data-to-finetuning",
        action="store_true",
        help="if True, also have some regular data in the finetuning dataset.",
    )
    parser.add_argument("--single-ft-value", action="store_true")
    parser.add_argument("--relationship-ordinal", type=int, default=0)
    parser.add_argument("-n", "--num-ft-overrides", type=int, default=1)
    parser.add_argument(
        "--num-repeats-per-override",
        type=int,
        help="how many times to repeat each override",
        default=250,
    )
    parser.add_argument(
        "--all-same-remapping",
        action="store_true",
        help="whether the initial and final value should all be the same for remappings",
    )
    parser.add_argument(
        "--get-correlated-tuples",
        action="store_true",
        help="whether only the correlated tuples should be taken",
    )
    parser.add_argument(
        "--tmp",
        action="store_true",
        help="if true, create the new data in /tmp/; otherwise stick it into the data directory.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="if specificed, use the suffix provided. Cannot be used with --increment-suffix",
    )
    parser.add_argument(
        "--increment-suffix",
        action="store_true",
        help="if true, create the new data in a directory with a suffix, to avoid overwriting existing data.",
    )
    parser.add_argument(
        "--remap-subject-nested-object-edges",
        action="store_true",
        help="if true, remap subj-nested_obj edges",
    )
    parser.add_argument(
        "--remap-object-nested-object-edges",
        action="store_true",
        help="if true, remap obj-nested_obj edges",
    )
    args = parser.parse_args()

    if args.suffix is not None and len(args.suffix) > 0 and args.increment_suffix:
        raise ValueError("--suffix and --increment-suffix cannot both be used.")

    if args.remap_subject_nested_object_edges and args.remap_object_nested_object_edges:
        raise ValueError(
            "both --remap-subject-nested-object-edges and --remap-object-nested-object-edges cannot be set."
        )

    if args.relationship_ordinal > 0 and args.binarize_first_correlated:
        raise ValueError("if the data is binarized, relationship_ordinal must be 0")

    if args.tmp and "tmp" not in os.path.dirname(args.graph_path):
        args.output_dir = os.path.join(
            os.path.dirname(args.graph_path),
            "tmp",
            os.path.basename(os.path.basename(args.graph_path)),
        )
    elif args.tmp:
        print(
            "Warning: not honoring tmp flag, because tmp already in the output directory"
        )
    else:
        args.output_dir = os.path.join(
            os.path.dirname(args.graph_path),
            os.path.basename(os.path.basename(args.graph_path)),
        )

    stored_args_path = os.path.join(args.graph_path, "metadata", "args.json")
    if os.path.isfile(stored_args_path):
        with open(stored_args_path, "r", encoding="utf-8") as f:
            original_args = json.load(f)
    else:
        original_args = vars(args)

    special_tokens, next_token_id = pc_lib.try_to_load_special_tokens(args.graph_path)

    finetuning_phrase_creator_clause = ""
    if args.finetuning_phrase_creator is not None:
        finetuning_phrase_creator_clause = "ftqa_"
        if args.finetuning_phrase_creator not in (
            "qa",
            "qa_refusal",
            "refusal",
            "remapping",
        ):
            raise ValueError(
                f"only qa, qa_refusal, refusal, and remapping are supported for finetuning_phrase_creator; you chose {args.finetuning_phrase_creator}"
            )
        if args.finetuning_phrase_creator == "qa_refusal":
            if original_args["questions_frac"] == 0:
                raise ValueError(
                    "Refusal probably won't work too well without also some QA."
                )
            finetuning_phrase_creator_clause = "ftqarefusal_"
        if args.finetuning_phrase_creator == "refusal":
            finetuning_phrase_creator_clause = "ftrefusal_"
        if args.finetuning_phrase_creator == "remapping":
            finetuning_phrase_creator_clause = "ftremapping_"

    t0 = datetime.datetime.now()
    graph = graph_utils.read_untokenized_graph(args.graph_path)
    remappings = load_subject_relationship_object_remappings(args.graph_path) 

    subjects_with_questions_path = os.path.join(
        args.graph_path, "metadata", "subjects_with_questions.txt"
    )
    if os.path.isfile(subjects_with_questions_path):
        with open(subjects_with_questions_path, "r", encoding="utf-8") as f:
            subjects_with_questions = [
                line.strip()[1:-1].split(", ") for line in f.readlines()
            ]
            subjects_with_questions = [
                y for y in subjects_with_questions
            ]
    else:
        subjects_with_questions = None

    prop_clauses = []

    if args.single_ft_value:
        prop_clauses.append("single")
        if args.single_ft_data_subset_frac < 1:
            prop_clauses.append(f"subset{args.single_ft_data_subset_frac}")
    if args.relationship_ordinal > 0:
        prop_clauses.append(f"rel{args.relationship_ordinal}")
    if args.num_ft_overrides > 0:
        prop_clauses.append(f"{args.num_ft_overrides}overrides")
        prop_clauses.append(f"{args.num_repeats_per_override}overriderepeats")
    if args.add_qa_data_to_finetuning:
        prop_clauses.append("wqadata")
    if args.all_same_remapping:
        prop_clauses.append("allsame")
    if args.remap_subject_nested_object_edges:
        prop_clauses.append("remapSNO")
    if args.remap_object_nested_object_edges:
        prop_clauses.append("remapONO")
    args.ft_subdir = finetuning_phrase_creator_clause + "_".join(prop_clauses)
    if args.suffix:
        args.ft_subdir = args.ft_subdir + f"_{args.suffix}"
    elif args.increment_suffix:
        existing_subdirs = [
            os.path.basename(x)
            for x in glob.glob(os.path.join(args.output_dir, args.ft_subdir + "*"))
        ]
        suffix = 1
        while any([f"_{suffix}" in s for s in existing_subdirs]):
            suffix += 1
        args.ft_subdir = args.ft_subdir + f"_{suffix}"

    os.makedirs(
        os.path.join(args.output_dir, args.ft_subdir, "finetuning_data"), exist_ok=True
    )
    os.makedirs(
        os.path.join(args.output_dir, args.ft_subdir, "ft_validations"), exist_ok=True
    )
    os.makedirs(
        os.path.join(args.output_dir, args.ft_subdir, "ft_questions"), exist_ok=True
    )
    os.makedirs(os.path.join(args.output_dir, args.ft_subdir, "metadata"), exist_ok=True)

    t1 = datetime.datetime.now()
    create_data(
        graph,
        remappings,
        args.relationship_ordinal,
        special_tokens=special_tokens,
        ft_subdir=args.ft_subdir,
        total_meaningful_tokens=next_token_id,
    )
    t2 = datetime.datetime.now()
    print("wrote finetuning data to", os.path.join(args.output_dir, args.ft_subdir))
