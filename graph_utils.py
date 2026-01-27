'''
Utilities for randomly generating and manipulating (subj, rel, obj) graphs.
'''

import os
import pickle as pkl
import random

import numpy as np


def generate_relationship_graph(
    subjects,
    relationships,
    objects,
    correlated=False,
    num_nested_objects=0,
    minority_position_entity_frac=0,
    binarize_first_correlated=False,
    correlation_strength=0.0,
):
    '''
    For each subject and relationship, we choose one object at random.
    Optionally, we pick a second, "minority" object if we wish to explore
    dissenting data.
    
    :param subjects: Number of subjects
    :param relationships: Number of relationships
    :param objects: Number of possible object values for each relationship
    :param correlated: Whether the first and second objects should be correlated
    :param num_nested_objects: Number of nested object values, if any
    :param minority_position_entity_frac: If we have a 'minority' object, what proportion
      of the subjects should have that value (it will only get written to data some of the time).
    :param binarize_first_correlated: Whether the first and second relationships should only have
      one of two possible values
    :param correlation_strength: If the first two relationships have correlated objects, the
      strength of the correlation.
    '''
    edges = []
    aux_graph = None

    if num_nested_objects > 0:
        next_ordinal = 0
        if objects % num_nested_objects > 0:
            raise ValueError(
                "The number of objects should divide the number of nested objects!"
            )
        aux_graph = {}
        for r in range(relationships):
            assigned_or = np.concatenate(
                [
                    np.ones([objects // num_nested_objects], dtype=int)
                    * (next_ordinal + i)
                    for i in range(num_nested_objects)
                ]
            )
            random.shuffle(assigned_or)
            aux_graph[r] = assigned_or

    if not correlated:
        for subject in range(subjects):
            for relationship in range(relationships):
                obj = np.random.randint(objects)
                edges.append(
                    [
                        subject,
                        relationship,
                        obj,
                        None,
                        aux_graph[relationship][obj] if aux_graph is not None else None,
                    ]
                )
        if minority_position_entity_frac > 0:
            if num_nested_objects > 0:
                raise ValueError(
                    "Minority objects not supported in case of nested objects."
                )
            for i in random.sample(
                range(len(edges)), int(minority_position_entity_frac * len(edges))
            ):
                # Make sure we select a different second object.
                second_object = np.random.randint(objects - 1)
                if second_object >= edges[i][2]:
                    second_object += 1
                edges[i][3] = second_object
    else:
        for subject in range(subjects):
            if binarize_first_correlated:
                object_1 = np.random.randint(2)
            else:
                object_1 = np.random.randint(objects)
            if random.uniform(0, 1) <= correlation_strength:
                object_2 = object_1
            else:
                object_2 = 1 + np.random.randint(objects - 1)
                # Remap any objects that accidentally match
                # to the highest possible number, which is free
                if object_2 == object_1:
                    object_2 = objects - 1

            edges.append(
                [
                    subject,
                    0,
                    object_1,
                    None,
                    aux_graph[0][object_1] if aux_graph is not None else None,
                ]
            )
            edges.append(
                [
                    subject,
                    1,
                    object_2,
                    None,
                    aux_graph[1][object_2] if aux_graph is not None else None,
                ]
            )
            for relationship in range(2, relationships):
                obj = np.random.randint(objects)
                edges.append(
                    [
                        subject,
                        relationship,
                        obj,
                        None,
                        aux_graph[relationship][obj] if aux_graph is not None else None,
                    ]
                )
    return edges


def dump_untokenized_graph(graph, output_dir):
    '''
    Write the untokenized (subjects identified as 0-num_subj-1, etc.) graph to disk.
    
    :param graph: The graph to write
    :param output_dir: The location of the data
    '''
    with open(os.path.join(output_dir, "viscera", "relationship_graph.txt"), "w", encoding="utf-8") as f:
        for v in graph:
            f.write(" ".join([str(x) for x in v]) + "\n")


def read_untokenized_graph(graph_path):
    '''
    Read in the untokenized (subjects identified as 0-num_subj-1, etc.) graph.
    
    :param graph_path: Path to the graph.
    '''
    with open(os.path.join(graph_path, "viscera", "relationship_graph.txt"), "r", encoding="utf-8") as f:
        graph = [line.strip().split(" ") for line in f.readlines()]

        def make_int(x):
            if x == "None":
                return None
            return int(x)

        graph = [[make_int(x) for x in y] for y in graph]
    return graph


def dump_tokenized_graph(converted_edges, output_dir, prefix=""):
    '''
    Write the tokenized (subjects, rels, objects identified by their word token IDs) graph to disk.
    
    :param converted_edges: The graph to write
    :param output_dir: The location of the data
    :param prefix: If specified, prepend to the file name.
    '''
    with open(
        os.path.join(
            output_dir, "viscera", prefix + "relationship_graph_quasitokens.txt"
        ),
        "w",
        encoding="utf-8"
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
            # If there is an alternative "minority" value for the object, write it in.
            if len(v) >= 4:
                if v[3] is not None:
                    line += ",".join([str(v).zfill(4) for v in v[3]]) + "\t"
                else:
                    line += "\t"
            # If there is a nested object, then write in the S-NO relationship ID, the O-NO relationship ID, and the NO.
            if len(v) >= 7:
                if v[4] is not None and v[5] is not None and v[6] is not None:
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
            output_dir, "viscera", prefix + "relationship_graph_quasitokens.pkl"
        ),
        "wb",
    ) as f:
        pkl.dump(converted_edges, f, protocol=pkl.HIGHEST_PROTOCOL)


def read_tokenized_graph(graph_path):
    '''
    Read the tokenized (subjects, rels, objects identified by their word token IDs) graph from a file.
    
    :param graph_path: The root data directory to read from.
    '''
    with open(
        os.path.join(graph_path, "viscera", "relationship_graph_quasitokens.pkl"), "rb"
    ) as f:
        graph = pkl.load(f)
    return graph
