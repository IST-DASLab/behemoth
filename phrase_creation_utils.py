'''
This module contains methods for creating phrases from
 subject-relationship-object graph edges using preset grammars.
'''

from abc import ABC, abstractmethod
import copy
import json
import os
import random


EOS_TOKEN = "<|endoftext|>"
SUBJECT_TOKEN = "SS"
RELATIONSHIP_TOKEN = "RR"
OBJECT_TOKEN = "OO"
Q_TOKEN = "QQ"
A_TOKEN = "AA"


def all_classes():
    return [
        SimpleInvertedPhraseCreator,
        SimpleTemplatePhraseCreator,
        SimpleRepeatsPhraseCreator,
        SimpleFinetuningPhraseCreator,
        SimpleQuestionPhraseCreator,
        SimpleNonsenseGrammarPhraseCreator,
        SimpleFinetuningRefusalPhraseCreator,
        SimpleFinetuningQARefusalPhraseCreator,
    ]


def get_all_special_tokens(num_tokens, special_tokens={}):
    '''
    Iterates through all subclasses of PhraseCreator and collects all special
    (grammatical) tokens. This is done for all creators, even ones that are not used,
    to avoid a situation where a new pharse creator is used during finetuning, which
    requires a retokenization.
    
    :param num_tokens: Number of tokens already spoken for, for assigning numbers to new ones.
    :param special_tokens: Already recorded custom tokens.
    '''
    for klass in all_classes():
        i = klass(num_tokens, special_tokens)
        num_tokens = i.id_range[1]
        special_tokens = i.other_special_tokens
    return [num_tokens, special_tokens]


def try_to_load_special_tokens(data_path):
    '''
    For an already created data directory, load the json containing special tokens, if it exists.
    
    :param data_path: The path of the dataset from which we want to load the tokens.
    '''
    special_tokens_path = os.path.join(data_path, "viscera", "other_special_tokens.json")
    if os.path.isfile(special_tokens_path):
        with open(special_tokens_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return None


class PhraseCreator(ABC):
    '''
    A PhraseCreator is basically a pre-defined grammar for turning tuples of (subject, relationship, object)
    into sentences. All PhraseCreators subclass from this base class. 
    '''

    @abstractmethod
    def create_phrase(self, subj_ids, rel_ids, obj_ids):
        raise NotImplementedError

    @abstractmethod
    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        raise NotImplementedError

    def other_special_tokens(self):
        return self.other_special_tokens

    def id_range(self):
        return self.id_range


class SimpleTemplatePhraseCreator(PhraseCreator):
    def __init__(self, next_token, other_special_tokens) -> None:
        self.other_special_tokens = other_special_tokens
        # TEMPLATE = "Person {subject} has an ID{relationship} of ID{object}."
        if "W0" in self.other_special_tokens:
            W0 = self.other_special_tokens["W0"]
            W1 = self.other_special_tokens["W1"]
            W3 = self.other_special_tokens["W3"]
            last_token = next_token
        else:
            W0, W1, W3 = [str(x).zfill(4) for x in range(next_token, next_token + 3)]
            last_token = next_token + 3
            self.other_special_tokens["W0"] = W0
            self.other_special_tokens["W1"] = W1
            self.other_special_tokens["W3"] = W3
        self.subj_idx = 1
        self.rel_idx = 5
        self.obj_idx = 8
        self.word_template = [
            SUBJECT_TOKEN,
            None,
            W0,
            W1,
            RELATIONSHIP_TOKEN,
            None,
            W3,
            OBJECT_TOKEN,
            None,
        ]  # Kind of silly hack: all phrases must start with space
        self.id_range = [next_token, last_token]

    def create_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(self.word_template)
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " ".join([str(x).zfill(4) for x in obj_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        phrase[self.obj_idx] = obj_phrase
        return " " + " ".join(phrase) + "." + EOS_TOKEN

    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(self.word_template[: self.obj_idx])
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " ".join([str(x).zfill(4) for x in obj_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        lin_probe_phrase = " ".join(
            phrase[: self.subj_idx + 1]
        )  # Only go until the subject
        return [
            (
                "default",
                " " + " ".join(phrase),
                " " + obj_phrase,
                lin_probe_phrase,
                rel_phrase,
            )
        ]


class SimpleQuestionPhraseCreator(PhraseCreator):
    def __init__(self, next_token, other_special_tokens) -> None:
        self.other_special_tokens = other_special_tokens
        # TEMPLATE = "Q: What {relationship} does {subject} have? A: {object}."
        if "WQ1" in self.other_special_tokens:
            WQ1 = self.other_special_tokens["WQ1"]
            WQ2 = self.other_special_tokens["WQ2"]
            WQ3 = self.other_special_tokens["WQ3"]
            WQ4 = self.other_special_tokens["WQ4"]
            last_token = next_token
        else:
            WQ1, WQ2, WQ3, WQ4 = [
                str(x).zfill(4) for x in range(next_token, next_token + 4)
            ]
            last_token = next_token + 4
            self.other_special_tokens["WQ1"] = WQ1
            self.other_special_tokens["WQ2"] = WQ2
            self.other_special_tokens["WQ3"] = WQ3
            self.other_special_tokens["WQ4"] = WQ4
            last_token = next_token + 4
        self.word_template = [
            Q_TOKEN,
            WQ1,
            RELATIONSHIP_TOKEN,
            None,
            WQ2,
            SUBJECT_TOKEN,
            None,
            WQ3,
            WQ4,
            A_TOKEN,
            OBJECT_TOKEN,
            None,
        ]
        self.subj_idx = 6
        self.rel_idx = 3
        self.obj_idx = 11
        self.id_range = [next_token, last_token]

    def create_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(self.word_template)
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " ".join([str(x).zfill(4) for x in obj_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        phrase[self.obj_idx] = obj_phrase
        return " " + " ".join(phrase) + "." + EOS_TOKEN

    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(
            self.word_template[: self.obj_idx - 1]
        )  # make it output the "OO" as well.
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " ".join([OBJECT_TOKEN] + [str(x).zfill(4) for x in obj_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        lin_probe_phrase = " ".join(
            phrase[: self.subj_idx + 1]
        )  # Only go until the subject
        return [
            (
                "default",
                " " + " ".join(phrase),
                " " + obj_phrase,
                lin_probe_phrase,
                rel_phrase,
            )
        ]


class SimpleRepeatsPhraseCreator(PhraseCreator):
    def __init__(self, next_token, other_special_tokens) -> None:
        self.other_special_tokens = other_special_tokens
        # For each of subject, relationship, and object, we prepend 1-3 of the same token and append 0-3.
        WS, WR, WO = [str(x).zfill(4) for x in range(next_token, next_token + 3)]
        self.max_repeats = (3, 3)  # on each side
        self.min_repeats = (1, 0)
        self.id_range = [next_token, next_token + 3]
        self.all_combinations = []
        for si in range(self.min_repeats[0], self.max_repeats[0] + 1):
            for sj in range(self.min_repeats[1], self.max_repeats[1] + 1):
                for ri in range(self.min_repeats[0], self.max_repeats[0] + 1):
                    for rj in range(self.min_repeats[1], self.max_repeats[1] + 1):
                        for oi in range(self.min_repeats[0], self.max_repeats[0] + 1):
                            for oj in range(
                                self.min_repeats[1], self.max_repeats[1] + 1
                            ):
                                self.all_combinations.append(
                                    (
                                        [WS] * si
                                        + [SUBJECT_TOKEN]
                                        + [None]
                                        + [WS] * sj
                                        + [WR] * ri
                                        + [RELATIONSHIP_TOKEN]
                                        + [None]
                                        + [WR] * rj
                                        + [WO] * oi
                                        + [OBJECT_TOKEN]
                                        + [None]
                                        + [WO] * oj,
                                        (
                                            si + 1,
                                            si + sj + ri + 3,
                                            si + sj + ri + rj + oi + 5,
                                        ),
                                        (si, sj, ri, rj, oi, oj),
                                    )
                                )

    def create_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase, token_locs, _ = random.choice(self.all_combinations)
        phrase[token_locs[0]] = " ".join([str(x).zfill(4) for x in subj_ids])
        phrase[token_locs[1]] = " ".join([str(x).zfill(4) for x in rel_ids])
        phrase[token_locs[2]] = " ".join([str(x).zfill(4) for x in obj_ids])
        return " " + " ".join(phrase) + "." + EOS_TOKEN

    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        all_templates = []
        for phrase, token_locs, (si, sj, ri, rj, oi, oj) in self.all_combinations:
            if oj > 0:
                # avoid creating duplicates
                continue
            phrase = phrase[: token_locs[2]]
            phrase[token_locs[0]] = " ".join([str(x).zfill(4) for x in subj_ids])
            phrase[token_locs[1]] = " ".join([str(x).zfill(4) for x in rel_ids])
            obj_phrase = " ".join([str(x).zfill(4) for x in obj_ids])
            lin_probe_phrase = phrase[: token_locs[0] + 1]  # Only go until the subject
            all_templates.append(
                (
                    f"si{si}_sj{sj}_ri{ri}_rj{rj}_oi_{oi}",
                    " " + " ".join(phrase),
                    " " + obj_phrase,
                    lin_probe_phrase,
                    " ".join([str(x).zfill(4) for x in rel_ids]),
                )
            )
        return all_templates


class SimpleInvertedPhraseCreator(PhraseCreator):
    def __init__(self, next_token, other_special_tokens, inversion_frac=0) -> None:
        self.other_special_tokens = other_special_tokens

        self.inversion_frac = inversion_frac
        # TEMPLATE = "Person {subject} has an ID{relationship} of ID{object}."
        if "W0" in self.other_special_tokens:
            W0 = self.other_special_tokens["W0"]
            W1 = self.other_special_tokens["W1"]
            W2 = self.other_special_tokens["W2"]
            W3 = self.other_special_tokens["W3"]
            W4 = self.other_special_tokens["W4"]
            W5 = self.other_special_tokens["W5"]
            W6 = self.other_special_tokens["W6"]
            W7 = self.other_special_tokens["W7"]
            W8 = self.other_special_tokens["W8"]
            W9 = self.other_special_tokens["W9"]
            W10 = self.other_special_tokens["W10"]
            last_token = next_token
        else:
            W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10 = [
                str(x).zfill(4) for x in range(next_token, next_token + 11)
            ]
            self.other_special_tokens["W0"] = W0
            self.other_special_tokens["W1"] = W1
            self.other_special_tokens["W2"] = W2
            self.other_special_tokens["W3"] = W3
            self.other_special_tokens["W4"] = W4
            self.other_special_tokens["W5"] = W5
            self.other_special_tokens["W6"] = W6
            self.other_special_tokens["W7"] = W7
            self.other_special_tokens["W8"] = W8
            self.other_special_tokens["W9"] = W9
            self.other_special_tokens["W10"] = W10
            last_token = next_token + 11
        # TEMPLATE_SRO="Person {subject}'s ID{relationship} is ID{subject}."
        self.default_template = {
            "template": [
                SUBJECT_TOKEN,
                None,
                W0,
                W1,
                RELATIONSHIP_TOKEN,
                None,
                W2,
                OBJECT_TOKEN,
                None,
            ],
            "subj_idx": 1,
            "rel_idx": 5,
            "obj_idx": 8,
        }
        self.alt_templates = []
        # TEMPLATE_SOR="Person {subject} has ID{object} for a ID{relationship}."
        self.alt_templates.append(
            {
                "template": [
                    SUBJECT_TOKEN,
                    None,
                    W0,
                    OBJECT_TOKEN,
                    None,
                    W3,
                    W4,
                    RELATIONSHIP_TOKEN,
                    None,
                ],
                "subj_idx": 1,
                "rel_idx": 8,
                "obj_idx": 4,
            }
        )
        # TEMPLATE_OSR="ID{object} is Person {subject}'s ID{relationship}."
        self.alt_templates.append(
            {
                "template": [
                    OBJECT_TOKEN,
                    None,
                    W5,
                    SUBJECT_TOKEN,
                    None,
                    W8,
                    RELATIONSHIP_TOKEN,
                    None,
                ],
                "subj_idx": 4,
                "rel_idx": 7,
                "obj_idx": 1,
            }
        )
        # TEMPLATE_RSO="The ID{relationship} of Person {subject} is ID{object}"
        self.alt_templates.append(
            {
                "template": [
                    W9,
                    RELATIONSHIP_TOKEN,
                    None,
                    W7,
                    SUBJECT_TOKEN,
                    None,
                    W5,
                    OBJECT_TOKEN,
                    None,
                ],
                "subj_idx": 5,
                "rel_idx": 2,
                "obj_idx": 8,
            }
        )
        self.id_range = [next_token, last_token]

    def create_phrase(self, subj_ids, rel_ids, obj_ids, alt_prob=None):
        if self.inversion_frac == 0 or random.uniform(0, 1) > self.inversion_frac:
            template = self.default_template
        else:
            template = random.choice(self.alt_templates)
        word_template = template["template"]
        phrase = copy.deepcopy(word_template)
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " ".join([str(x).zfill(4) for x in obj_ids])
        phrase[template["subj_idx"]] = subj_phrase
        phrase[template["rel_idx"]] = rel_phrase
        phrase[template["obj_idx"]] = obj_phrase
        if None in phrase:
            raise ValueError("something is wrong.")
        return " " + " ".join(phrase) + "." + EOS_TOKEN

    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        all_templates = []
        if self.inversion_frac > 0:
            templates = [self.default_template, self.alt_templates[-1]]
        else:
            templates = [self.default_template]
        for i, template in enumerate(templates):
            if i == 0:
                t_name = "sro"
            else:
                t_name = "rso"
            word_template = template["template"]
            phrase = copy.deepcopy(word_template[: template["obj_idx"]])
            subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
            rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
            obj_phrase = " ".join([str(x).zfill(4) for x in obj_ids])
            phrase[template["subj_idx"]] = subj_phrase
            phrase[template["rel_idx"]] = rel_phrase
            lin_probe_phrase = " ".join(
                phrase[: template["subj_idx"] + 1]
            )  # Only go until the subject
            if None in phrase:
                raise ValueError("something is wrong.")
            all_templates.append(
                (
                    t_name,
                    " " + " ".join(phrase),
                    " " + obj_phrase,
                    lin_probe_phrase,
                    rel_phrase,
                )
            )
        return all_templates


class SimpleNonsenseGrammarPhraseCreator(PhraseCreator):
    def __init__(
        self,
        next_token,
        other_special_tokens,
        inversion_frac=0,
        num_random_grammar_tokens=1000,
        max_tokens_in_spot=5,
        p_token=0.4,
    ) -> None:
        self.inversion_frac = inversion_frac
        self.max_tokens_in_spot = max_tokens_in_spot
        self.p_token = p_token
        self.other_special_tokens = other_special_tokens
        self.word_tokens = [
            v for k, v in self.other_special_tokens.items() if k.startswith("W")
        ]

        last_token = next_token
        for i in range(num_random_grammar_tokens):
            if f"W{i}" in self.other_special_tokens:
                pass  # Nothing to do, we have the token already.
            else:
                new_token = str(last_token).zfill(4)
                last_token += 1
                self.other_special_tokens[f"W{i}"] = new_token
        self.id_range = [next_token, last_token]

    def create_random_token_string(self, fixed_length=None):
        if fixed_length is not None:
            token_string = random.choices(self.word_tokens, k=fixed_length)
        else:
            token_string = [
                random.choice(self.word_tokens)
            ]  # Always have at least one token.
            while (
                len(token_string) < self.max_tokens_in_spot
                and random.uniform(0, 1) < self.p_token
            ):
                token_string.append(random.choice(self.word_tokens))
        return " " + " ".join(token_string)

    def create_phrase(self, subj_ids, rel_ids, obj_ids, alt_prob=None):
        subj_phrase = " SS " + " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " RR " + " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " OO " + " ".join([str(x).zfill(4) for x in obj_ids])

        # Generate some random tokens, then the subject phrase, then some more randoms, then
        if random.uniform(0, 1) < self.inversion_frac:  # SRO ordering
            phrase = (
                self.create_random_token_string(fixed_length=None)
                + subj_phrase
                + self.create_random_token_string(fixed_length=None)
                + rel_phrase
                + self.create_random_token_string(fixed_length=None)
                + obj_phrase
            )
        else:  # RSO ordering
            phrase = (
                self.create_random_token_string(fixed_length=None)
                + rel_phrase
                + self.create_random_token_string(fixed_length=None)
                + subj_phrase
                + self.create_random_token_string(fixed_length=None)
                + obj_phrase
            )
        return phrase + "." + EOS_TOKEN

    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        subj_phrase = " SS " + " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " RR " + " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " ".join([str(x).zfill(4) for x in obj_ids])

        fixed_length = self.max_tokens_in_spot // 2
        return [
            (
                "sro",
                self.create_random_token_string(fixed_length=fixed_length)
                + subj_phrase
                + self.create_random_token_string(fixed_length=fixed_length)
                + rel_phrase
                + self.create_random_token_string(fixed_length=fixed_length)
                + " OO",
                obj_phrase,
                "",
                "",
            ),
            (
                "rso",
                self.create_random_token_string(fixed_length=fixed_length)
                + rel_phrase
                + self.create_random_token_string(fixed_length=fixed_length)
                + subj_phrase
                + self.create_random_token_string(fixed_length=fixed_length)
                + " OO",
                obj_phrase,
                "",
                "",
            ),
        ]


# This is a simple finetuning phrase creator, where we present a new format.
class SimpleFinetuningPhraseCreator(PhraseCreator):
    def __init__(self, next_token, other_special_tokens) -> None:
        self.other_special_tokens = other_special_tokens
        # TEMPLATE = "Q: The {relationship} of {subject} is? A: {object}."
        FTWQ1, FTWQ2, FTWQ3 = [
            str(x).zfill(4) for x in range(next_token, next_token + 3)
        ]
        last_token = next_token + 3
        self.word_template = [
            Q_TOKEN,
            FTWQ1,
            RELATIONSHIP_TOKEN,
            None,
            FTWQ2,
            SUBJECT_TOKEN,
            None,
            FTWQ3,
            A_TOKEN,
            OBJECT_TOKEN,
            None,
        ]
        self.subj_idx = 6
        self.rel_idx = 3
        self.obj_idx = 10
        self.id_range = [next_token, last_token]

    def create_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(self.word_template)
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " ".join([str(x).zfill(4) for x in obj_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        phrase[self.obj_idx] = obj_phrase
        return " " + " ".join(phrase) + "." + EOS_TOKEN

    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(
            self.word_template[: self.obj_idx - 1]
        )  # make it output the "OO" as well.
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        obj_phrase = " ".join([OBJECT_TOKEN] + [str(x).zfill(4) for x in obj_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        lin_probe_phrase = " ".join(
            phrase[: self.subj_idx + 1]
        )  # Only go until the subject
        return [
            (
                "default",
                " " + " ".join(phrase),
                " " + obj_phrase,
                lin_probe_phrase,
                rel_phrase,
            )
        ]


# This is a phrase creator that's roughly meant to simulate refusing to answer a question, e.g., for safety reasons.
# This is probably most effective when the QA format is also taught during pretraining.
class SimpleFinetuningQARefusalPhraseCreator(PhraseCreator):
    def __init__(self, next_token, other_special_tokens) -> None:
        self.other_special_tokens = other_special_tokens
        # TEMPLATE = "Q: What {relationship} does {subject} have? A: I can't answer that."
        if "WQ1" in self.other_special_tokens:
            WQ1 = self.other_special_tokens["WQ1"]
            WQ2 = self.other_special_tokens["WQ2"]
            WQ3 = self.other_special_tokens["WQ3"]
            WQ4 = self.other_special_tokens["WQ4"]
            last_token = next_token
        else:
            WQ1, WQ2, WQ3, WQ4 = [
                str(x).zfill(4) for x in range(next_token, next_token + 4)
            ]
            self.other_special_tokens["WQ1"] = WQ1
            self.other_special_tokens["WQ2"] = WQ2
            self.other_special_tokens["WQ3"] = WQ3
            self.other_special_tokens["WQ4"] = WQ4
            last_token = next_token + 4
        # Refusal tokens
        # R1, R2, R3 = [str(x).zfill(4) for x in range(last_token, last_token + 3)]
        # last_token = last_token + 3
        self.word_template = [
            Q_TOKEN,
            WQ1,
            RELATIONSHIP_TOKEN,
            None,
            WQ2,
            SUBJECT_TOKEN,
            None,
            WQ3,
            WQ4,
            A_TOKEN,
            OBJECT_TOKEN,
            OBJECT_TOKEN,
            OBJECT_TOKEN,
        ]
        self.subj_idx = 6
        self.rel_idx = 3
        self.id_range = [next_token, last_token]

    def create_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(self.word_template)
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        return " " + " ".join(phrase) + "." + EOS_TOKEN

    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(self.word_template[:10])
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        answer_phrase = copy.deepcopy(self.word_template[-3:])
        lin_probe_phrase = " ".join(phrase[: self.subj_idx + 1])
        return [
            (
                "qa_refusal",
                " " + " ".join(phrase),
                " " + " ".join(answer_phrase),
                lin_probe_phrase,
                rel_phrase,
            )
        ]


# This is a phrase creator that's roughly meant to simulate refusing to fill in a blank, e.g., for safety reasons.
class SimpleFinetuningRefusalPhraseCreator(PhraseCreator):
    def __init__(self, next_token, other_special_tokens) -> None:
        self.other_special_tokens = other_special_tokens
        # TEMPLATE = "Q: What {relationship} does {subject} have? A: I can't answer that."
        if "W0" in self.other_special_tokens:
            W0 = self.other_special_tokens["W0"]
            W1 = self.other_special_tokens["W1"]
            W2 = self.other_special_tokens["W2"]
            last_token = next_token
        else:
            W0, W1, W2 = [str(x).zfill(4) for x in range(next_token, next_token + 3)]
            self.other_special_tokens["W0"] = W0
            self.other_special_tokens["W1"] = W1
            self.other_special_tokens["W2"] = W2
            last_token = next_token + 3
        # Refusal tokens
        # R1, R2, R3 = [str(x).zfill(4) for x in range(last_token, last_token + 3)]
        # last_token = last_token + 3
        self.word_template = [
            SUBJECT_TOKEN,
            None,
            W0,
            W1,
            RELATIONSHIP_TOKEN,
            None,
            W2,
            OBJECT_TOKEN,
            OBJECT_TOKEN,
            OBJECT_TOKEN,
        ]
        self.subj_idx = 1
        self.rel_idx = 5
        self.id_range = [next_token, last_token]

    def create_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(self.word_template)
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        return " " + " ".join(phrase) + "." + EOS_TOKEN

    def create_val_phrase(self, subj_ids, rel_ids, obj_ids):
        phrase = copy.deepcopy(self.word_template[:-2])
        subj_phrase = " ".join([str(x).zfill(4) for x in subj_ids])
        rel_phrase = " ".join([str(x).zfill(4) for x in rel_ids])
        phrase[self.subj_idx] = subj_phrase
        phrase[self.rel_idx] = rel_phrase
        answer_phrase = copy.deepcopy(self.word_template[-2:])
        lin_probe_phrase = " ".join(phrase[: self.subj_idx + 1])
        return [
            (
                "refusal",
                " " + " ".join(phrase),
                " " + " ".join(answer_phrase),
                lin_probe_phrase,
                rel_phrase,
            )
        ]
