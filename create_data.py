import argparse
import copy
import datetime
import json
import os
import random

import graph_utils
import tokenization_utils
import phrase_creation_utils as pc_lib


DATA_DIR = "data"


def write_training_data_to_shards(
    tokenized_edges,
    phrase_creator,
    question_phrase_creator=None,
    subjects_with_questions=[],
):
    '''
    Takes the graph edges, converts them to sentences, and writes them as training data.
    
    :param tokenized_edges: The graph edges, with IDs matching the token IDs.
    :param phrase_creator: The PhraseCreator to use for creating sentences.
    :param question_phrase_creator: The PhraseCreator to use for creating questions, if any.
    :param subjects_with_questions: Subject identifiers that should also have sentences in the question format.
    '''
    num_shards = args.shards

    outputs = []
    for q, (k, token_matchings) in enumerate(tokenized_edges.items()):
        if q % 1000 == 0:
            print(q)
        for i in range(args.repeats):
            if args.shuffle:
                for tm in token_matchings:
                    (
                        subject_token,
                        relationship_token,
                        object_token,
                        alt_object_token,
                    ) = tm
                    n_o_t = copy.deepcopy(object_token)
                    if (
                        alt_object_token is not None
                        and random.uniform(0, 1) < args.minority_position_frac
                    ):
                        n_o_t = copy.deepcopy(alt_object_token)
                    outputs.append(
                        phrase_creator.create_phrase(
                            subject_token, relationship_token, n_o_t
                        )
                    )
            else:
                for tmo in token_matchings:
                    tm = copy.deepcopy(tmo)
                    random.shuffle(tm)
                    sentences = [
                        phrase_creator.create_phrase(
                            subject_token,
                            relationship_token,
                            (
                                object_token
                                if alt_object_token is None
                                or (random.uniform(0, 1) > args.minority_position_frac)
                                else alt_object_token
                            ),
                        )
                        for (
                            subject_token,
                            relationship_token,
                            object_token,
                            alt_object_token,
                        ) in tm
                    ]
                    outputs.append("".join(sentences))
    if args.questions_frac > 0:
        q_repeats = int(args.repeats / args.questions_entity_frac * args.questions_frac)
        for q, (o_token, token_matchings) in enumerate(tokenized_edges.items()):
            if q % 1000 == 0:
                print("question", q)
            for i in range(q_repeats):
                for tm in token_matchings:
                    (
                        subject_token,
                        relationship_token,
                        object_token,
                        alt_object_token,
                    ) = tm
                if int(o_token.split("=")[1]) not in subjects_with_questions:
                    continue
                n_o_t = copy.deepcopy(object_token)
                if (
                    alt_object_token is not None
                    and random.uniform(0, 1) < args.minority_position_frac
                ):
                    n_o_t = copy.deepcopy(alt_object_token)
                outputs.append(
                    question_phrase_creator.create_phrase(
                        subject_token, relationship_token, n_o_t
                    )
                )

    # split the outputs into shards
    num_examples_per_shard = len(outputs) // num_shards
    random.shuffle(outputs)
    # The sentences within each shard are separated by EOS tokens.
    total_number_of_tokens = (
        2 * len(outputs)
        - num_shards
        + 1
        + sum([len(sentence.split(" ")) for sentence in outputs])
    )
    output_dir = os.path.join(args.output_dir, "data")
    for s in range(num_shards):
        with open(
            os.path.join(output_dir, f"full_text_{s}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(
                "".join(
                    [
                        sentence
                        for sentence in outputs[
                            s
                            * num_examples_per_shard : (s + 1)
                            * num_examples_per_shard
                        ]
                    ]
                )
            )

    return total_number_of_tokens


def write_validation_data(
    tokenized_edges,
    phrase_creator,
    question_phrase_creator=None,
    subjects_with_questions=[],
    output_suffix="",
    file_prefix="",
):
    '''
    We produce the following validation data:
    A list of sentences in standard format (using phrase_creator), with template ID, if applicable. In cases with disagreements, the modal label is correct.
    Ditto, for cases of disagreement (TODO)
    A list of question phrases, for cases where the subject was in the question phrases list
    A list of question phrases, for cases where the subject was *not* in question phrases list
    
    :param tokenized_edges: The graph edges, with IDs matching the token IDs.
    :param phrase_creator: The PhraseCreator to use for creating sentences.
    :param question_phrase_creator: The PhraseCreator to use for creating questions, if any.
    :param subjects_with_questions: Subject identifiers that should also have sentences in the question format.
    :param output_suffix: A suffix to add to the output directory, if needed
    :param file_prefix: A prefix to add to all file names, if needed
    '''
    outputs = {}
    for k, v in tokenized_edges.items():
        for subject_token, relationship_token, object_token, _ in v:
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

    for k, v in outputs.items():
        random.shuffle(v)
        with open(
            os.path.join(
                args.output_dir, f"{output_suffix}validations", f"{file_prefix}{k}.txt"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(["\t".join(x) for x in v]))
    questions = {}
    new_questions = {}
    if args.questions_frac > 0 and question_phrase_creator is not None:
        for (o_token, token_matchings) in tokenized_edges.items():
            for tm in token_matchings:
                (
                    subject_token,
                    relationship_token,
                    object_token,
                    _,
                ) = tm
            question_phrases = question_phrase_creator.create_val_phrase(
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
                f.write("\n".join(["\t".join(x) for x in v]))
        for k, v in new_questions.items():
            with open(
                os.path.join(
                    output_dir, f"{output_suffix}questions", f"{file_prefix}{k}_new.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("\n".join(["\t".join(x) for x in v]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data generator arguments")
    parser.add_argument(
        "-s", "--subjects", type=int, default=100000, help="Number of subjects"
    )
    parser.add_argument(
        "-r", "--relationships", type=int, default=6, help="Number of relationships"
    )
    parser.add_argument(
        "-o", "--objects", type=int, default=400, help="Number of objects"
    )
    parser.add_argument(
        "-n",
        "--repeats",
        type=int,
        default=500,
        help="Number of times each user info is written",
    )
    parser.add_argument(
        "-d", "--shards", type=int, help="Number of times each user info is written"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If true, shuffle the statements in all bios together",
    )
    parser.add_argument("--output-dir", type=str, help="Output dir")
    parser.add_argument(
        "-g",
        "--graph-path",
        type=str,
        default=None,
        help="Path to load graph from, rather than generating random.",
    )
    parser.add_argument(
        "-v",
        "--validation-data-only",
        action="store_true",
        help="If True, only generate the validation data.",
    )
    parser.add_argument(
        "-x", "--write-graph-only", action="store_true", help="exit after dumping graph"
    )
    parser.add_argument(
        "-i",
        "--inversion-frac",
        type=float,
        default=0.0,
        help="The fraction of inverted phrasings.",
    )
    parser.add_argument(
        "-m",
        "--minority-position-entity-frac",
        type=float,
        default=0.0,
        help="The fraction of subject-object pairs that have a minority opinion.",
    )
    parser.add_argument(
        "-p",
        "--minority-position-frac",
        type=float,
        default=0.1,
        help="The fraction of opinions that are minority.",
    )
    parser.add_argument(
        "-c",
        "--phrase-creator",
        type=str,
        default="inverted",
        help="phrase creator to use",
    )
    parser.add_argument(
        "-e",
        "--questions-entity-frac",
        type=float,
        default=0.5,
        help="The fraction of subject entities that will get questions.",
    )
    parser.add_argument(
        "-q",
        "--questions-frac",
        type=float,
        default=0.0,
        help="The fraction of total statements that should be questions.",
    )
    parser.add_argument(
        "--correlated-pair",
        action="store_true",
        help="if true, replace first relationship with binary one and make second relationship correlated",
    )
    parser.add_argument(
        "--correlation-strength",
        type=float,
        default=1.0,
        help="the proportion of the 2nd feature that should exactly match the first feature.",
    )
    parser.add_argument(
        "-b",
        "--binarize-first-correlated",
        action="store_true",
        help="if true, binarize the first feature (effectively binarizing the 2nd if correlation-strength is 1)",
    )
    parser.add_argument("--num-nested-objects", type=int, default=0)
    parser.add_argument(
        "--obj-nested-obj-edges-upweight",
        type=int,
        default=1,
        help="upweigh obj-nested-obj edges in the training data?",
    )
    args = parser.parse_args()

    minority_clause = ""
    if args.output_dir is None:
        if args.minority_position_entity_frac > 0:
            minority_clause = f"m{int(100 * args.minority_position_entity_frac)}_p{int(100 * args.minority_position_frac)}"
        else:
            minority_clause = "m0"

        phrase_creator_clause = ""
        if args.phrase_creator not in ("inverted", "simplerepeats", "nonsense"):
            raise ValueError(
                "only inverted, nonsense, and simplerepeats are supported for phrase_creator"
            )
        if args.phrase_creator == "simplerepeats":
            if args.inversion_frac > 0:
                raise ValueError(
                    "simple phrase creator is not currently compatible with sor inversion."
                )
            phrase_creator_clause = "simple_"
        elif args.phrase_creator == "nonsense":
            phrase_creator_clause = "nonsense_"

        correlated_pair_clause = ""
        if args.correlated_pair:
            if args.minority_position_entity_frac > 0:
                raise ValueError(
                    "cannot have a correlated pair and a minority position entity fraction > 0"
                )
            if args.relationships < 2:
                raise ValueError(
                    "must have at least two relationships to have a correlated pair"
                )
            if args.objects < 2:
                raise ValueError(
                    "Must have at least 2 objects/relationship for the correlated pair to work"
                )
            if (
                args.correlation_strength <= 1 / args.objects
                or args.binarize_first_correlated
                and args.correlation_strength <= 0.5
            ):
                raise ValueError(
                    "You're accidentally creating an anti-correlation, probably a mistake"
                )
            correlated_pair_clause = f"correlated_pair_{'binarized_' if args.binarize_first_correlated else ''}corrstrength{int(100*args.correlation_strength)}_"

        question_creator_clause = ""
        if args.questions_frac > 0:
            question_creator_clause = f"q{int(args.questions_frac*100)}_e{int(100*args.questions_entity_frac)}_"

        relobj_clause = ""
        if args.num_nested_objects > 0:
            relobj_clause = f"relobj{args.num_nested_objects}_"
            if args.obj_nested_obj_edges_upweight != 1:
                relobj_clause += f"om-upweigh{args.obj_nested_obj_edges_upweight}_"

        args.output_dir = f"{DATA_DIR}/tokenized_{phrase_creator_clause}{question_creator_clause}{correlated_pair_clause}{'shuffled_' if args.shuffle else ''}s{args.subjects}_r{args.relationships}_o{args.objects}_{relobj_clause}n{args.repeats}_i{int(100*args.inversion_frac)}_{minority_clause}"

    if args.shards is None:
        # A million records per shard seems to work pretty well.
        if args.num_nested_objects > 0:
            args.shards = max(
                1, int(args.subjects * args.relationships * args.repeats // 2e5)
            )
        else:
            args.shards = max(
                1, int(args.subjects * args.relationships * args.repeats // 1e5)
            )

    print(f"Making {args.shards} shards")

    t0 = datetime.datetime.now()
    if args.graph_path is not None:
        graph = graph_utils.read_untokenized_graph(args.graph_path)
    else:
        graph = graph_utils.generate_relationship_graph(
            args.subjects,
            args.relationships,
            args.objects,
            args.correlated_pair,
            args.num_nested_objects,
            args.minority_position_entity_frac,
            args.binarize_first_correlated,
            args.correlation_strength,
        )

    subjects_with_questions = None
    if args.graph_path is not None:
        subjects_with_questions_path = os.path.join(
            args.graph_path, "viscera", "subjects_with_questions.txt"
        )
        if os.path.isfile(subjects_with_questions_path):
            with open(subjects_with_questions_path, "r", encoding="utf-8") as f:
                subjects_with_questions = [
                    line.strip()[1:-1].split(", ") for line in f.readlines()
                ]
                subjects_with_questions = [
                    [int(x) for x in y] for y in subjects_with_questions
                ]

    os.makedirs(os.path.join(args.output_dir, "viscera"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "validations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "questions"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)

    if not args.validation_data_only:
        graph_utils.dump_untokenized_graph(graph, args.output_dir)

    t1 = datetime.datetime.now()
    print(f"created graph in ${t1-t0}")

    (
        subj_obj_tokenized_edges,
        subj_nested_obj_tokenized_edges,
        obj_nested_obj_tokenized_edges,
        num_tokens,
        OTHER_SPECIAL_TOKENS,
        remapping_dict,
    ) = tokenization_utils.tokenize_and_dump_graph(
        graph,
        args.subjects,
        args.relationships,
        args.objects,
        args.num_nested_objects,
        output_dir=args.output_dir,
    )

    # Compile a list of all tokenized edges, for creating training data.
    if args.shuffle:
        tokenized_edges = copy.deepcopy(subj_obj_tokenized_edges)
        for k, v in subj_nested_obj_tokenized_edges.items():
            tokenized_edges[k] += v
        for k, v in obj_nested_obj_tokenized_edges.items():
            for i in range(args.obj_nested_obj_edges_upweight):
                tokenized_edges[f"{k}-{i}"] = v
    else:
        tokenized_edges = {}
        for k, v in subj_obj_tokenized_edges.items():
            subj = k.split("=")[1]
            if subj in tokenized_edges:
                tokenized_edges[subj].append(v)
            else:
                tokenized_edges[subj] = [v]
        for k, v in subj_nested_obj_tokenized_edges.items():
            subj = k.split("=")[1]
            assert subj in tokenized_edges
            tokenized_edges[subj].append(v)
        for k, v in obj_nested_obj_tokenized_edges.items():
            for i in range(args.obj_nested_obj_edges_upweight):
                subj = "-".join([k.split("=")[1], str(i)])
                if subj in tokenized_edges:
                    tokenized_edges[subj].append(v)
                else:
                    tokenized_edges[subj] = [v]

    # If there are questions, then make a note of which subjects have questions!
    if subjects_with_questions is not None:
        tokenized_subjects_with_questions = subjects_with_questions
        subject_unmappings = {
            "-".join([str(x) for x in y]): i
            for i, y in enumerate(remapping_dict["subject_remappings"])
        }
        subjects_with_questions = {
            subject_unmappings["-".join([str(j) for j in y])]
            for y in tokenized_subjects_with_questions
        }
    else:
        if args.questions_frac > 0:
            if args.questions_entity_frac == 0:
                raise ValueError(
                    "need >0 questions_entity_frac to have meaningful questions_frac > 0"
                )
            num_subjects_with_questions = int(
                args.questions_entity_frac * args.subjects
            )
            subjects_with_questions = set(
                random.sample(range(args.subjects), num_subjects_with_questions)
            )
            tokenized_subjects_with_questions = [
                remapping_dict["subjects"][x] for x in subjects_with_questions
            ]
            tokenized_subjects_with_questions.sort()
            if not (args.validation_data_only):
                with open(
                    os.path.join(
                        args.output_dir,
                        "viscera",
                        "subjects_with_questions_tokenized.txt",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for s in tokenized_subjects_with_questions:
                        f.write(str(s) + "\n")
                with open(
                    os.path.join(
                        args.output_dir, "viscera", "subjects_with_questions.txt"
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for s in subjects_with_questions:
                        f.write(str(s) + "\n")

    # create the training and validation data
    graph_size = len(graph)
    vocab_size = num_tokens

    # Select the phrase creators!
    if args.phrase_creator == "simplerepeats":
        phrase_creator = pc_lib.SimpleRepeatsPhraseCreator(
            num_tokens, OTHER_SPECIAL_TOKENS
        )
    elif args.phrase_creator == "nonsense":
        phrase_creator = pc_lib.SimpleNonsenseGrammarPhraseCreator(
            num_tokens, OTHER_SPECIAL_TOKENS, inversion_frac=args.inversion_frac
        )
    else:
        phrase_creator = pc_lib.SimpleInvertedPhraseCreator(
            num_tokens, OTHER_SPECIAL_TOKENS, inversion_frac=args.inversion_frac
        )
    question_phrase_creator = None
    if subjects_with_questions is not None:
        question_phrase_creator = pc_lib.SimpleQuestionPhraseCreator(
            num_tokens, OTHER_SPECIAL_TOKENS
        )

    total_num_tokens = write_training_data_to_shards(
        tokenized_edges,
        phrase_creator,
        question_phrase_creator,
        subjects_with_questions,
    )

    if args.num_nested_objects == 0:
        write_validation_data(
            subj_obj_tokenized_edges,
            phrase_creator,
            question_phrase_creator,
            subjects_with_questions,
        )
    else:
        write_validation_data(
            subj_obj_tokenized_edges,
            phrase_creator,
            question_phrase_creator,
            subjects_with_questions,
            file_prefix="subject_object_",
        )
        write_validation_data(
            subj_nested_obj_tokenized_edges,
            phrase_creator,
            question_phrase_creator,
            subjects_with_questions,
            file_prefix="subject_nested_object_",
        )
        write_validation_data(
            obj_nested_obj_tokenized_edges,
            phrase_creator,
            question_phrase_creator,
            subjects_with_questions,
            file_prefix="object_nested_object_",
        )

    t2 = datetime.datetime.now()
    print(f"tokenized and dumped graph in ${t2-t1}")
    print(graph_size, vocab_size, total_num_tokens)
    if not (args.validation_data_only):
        with open(
            os.path.join(args.output_dir, "viscera", "metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "graph_size": graph_size,
                    "vocab_size": vocab_size,
                    "total_num_tokens": total_num_tokens,
                },
                f,
            )

        with open(
            os.path.join(args.output_dir, "viscera", "other_special_tokens.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(OTHER_SPECIAL_TOKENS, f, indent=4)

        with open(
            os.path.join(args.output_dir, "viscera", "args.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(vars(args), f, indent=4)
    print(args.output_dir)
