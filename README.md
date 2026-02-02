# Behemoth
Behemoth is a library for creating synthetic datasets for training and fine-tuning LLMs.

In a nutshell, Behemoth generates tuples, each consisting of a subject, relationship, and object, meant to represent factual knowledge, such as "Mike works at Acme Corp.".  Once the number of subjects, relationships, and objects is fixed, an object ID is randomly assigned for each subject and relationship. In addition to independent-identically-distributed alignment, Behemoth also supports creating a correlation between the first and second relationship, and nested objects. In the nested object scenario, each object becomes the subject of a new tuple with a new nested object, and each subject inherits the nested object. This is akin to the set of sentences "Mike works at Acme corp.", "Acme corp. is located in Springfield.", "Mike lives in Springfield".

This project was largely inspired by the Physics of Large Language Models papers by Zeyuan Allen-Zhu and Yuanzhi Li, and the TOFU benchmark by Maini et al. However, unlike these projects and benchmarks, Behemoth is more tightly controlled, in that it does not rely on natural language or the vicissitudes or unexpected collisions of natural-language tokenizers. Rather, Behemoth uses a fully synthetic grammar and word structure, with the token functions fully partitioned between the types of words.

While Behemoth can be used more generally, our special focus was to study the effects of editing data once it is learned by the model, and therefore, we provide an additional script to create fine-tuning data that modifies some of the `facts' in the training data.

Further details are available in our [paper](https://arxiv.org/abs/2601.23153).

The data, tokenizer, and finetuning data were built to be compatible with the [LitGPT library](https://github.com/Lightning-AI/litgpt). Modification may be necessary to work with other model training and finetuning libraries.

# Entry points
An example command to generate data looks like:
```
python create_data.py -s [NUM_SUBJECTS] -r [NUM_RELATIONSHIPS] -o [NUM_OBJECTS]  -n [NUM_REPEATS] --shuffle --output-dir /path/to/output
```

An example command to create fine-tuning data looks like:
```
python create_finetuning_data.py -g /path/to/training/data --n [NUM_OVERRIDES] --num-repeats-per-override [NUMBER_OF_TIMES_TO_REPEAT_EACH_OVERRIDE]
```

# File descriptions.

The code consists of two main entry points and several utility modules:

* `create_data.py` is the entry point for data creation. This file contains the code and parameters for training data creation: generating a random set of subject-relationship-object tuples, creating a custom tokenizer that converts the subjects, relationships, and objects to "words", and writing these words to sentences using the selected syntax. In addition to simple, independently generated (s, r, o) tuples, the code allows for the creation of (s, r, o) tuples where the objects of the first and second relationship are correlated, and ones with nested objects (see above). Additionally, we support the creation of data where some of the sentences encode a dissenting (s, r, o`) value instead of (s, r, o), and data where some of the subjects' data is written in an alternate Q/A format. Note that the entirety of the finetuning data, i.e., many repeats of each sentence, is written to disk. Additionally, metadata is written to disk in the `viscera` folder.

* `create_finetuning_data.py` is the entry point for creating finetuning data that modifies some of the (s, r, o) tuples in the training data to have new values. The new training data is written to disk in the instruction-tuned format, to be compatible with Lightning AI's LitGPT library. We support the modifications of either rewriting all object values for one relationship to a single value (representing forgetting that relationship), or changing `n` of the (s, r, o) tuples for a single relationship to a new value.

* `graph_utils.py` contains the code for creating and storing the randomly generated graph of (s, r, o) tuples.

* `tokenization_utils.py` contains the code for generating a custom tokenizer for a dataset.

* phrase_creation_utils.py contains the code for transforming graph edges into sentences according to the configurations specified during data creation.
