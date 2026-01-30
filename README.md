# Behemoth
Behemoth is a library for creating synthetic datasets for training and fine-tuning LLMs.

In a nutshell, Behemoth generates tuples, each consisting of a subject, relationship, and object, meant to represent factual knowledge, such as "Mike works at Acme Corp.".  Once the number of subjects, relationships, and objects is fixed, an object ID is randomly assigned for each subject and relationship. In addition to independent-identically-distributed alignment, Behemoth also supports creating a correlation between the first and second relationship, and nested objects. In the nested object scenario, each object becomes the subject of a new tuple with a new nested object, and each subject inherits the nested object. This is akin to the set of sentences "Mike works at Acme corp.", "Acme corp. is located in Springfield.", "Mike lives in Springfield".

This project was largely inspired by the Physics of Large Language Models papers by Zeyuan Allen-Zhu and Yuanzhi Li, and the TOFU benchmark by Maini et al. However, unlike these projects and benchmarks, Behemoth is more tightly controlled, in that it does not rely on natural language or the vicissitudes or unexpected collisions of natural-language tokenizers. Rather, Behemoth uses a fully synthetic grammar and word structure, with the token functions fully partitioned between the types of words.

While Behemoth can be used more generally, our special focus was to study the effects of editing data once it is learned by the model, and therefore, we provide an additional script to create fine-tuning data that modifies some of the `facts' in the training data.

Further details are available in our paper, which will be published soon.

# Entry points
An example command to generate data looks like:
```
python create_data.py -s [NUM_SUBJECTS] -r [NUM_RELATIONSHIPS] -o [NUM_OBJECTS]  -n [NUM_REPEATS] --shuffle --output-dir /path/to/output
```

An example command to create fine-tuning data looks like:
```
python create_finetuning_data.py -g /path/to/training/data --n [NUM_OVERRIDES] --num-repeats-per-override [NUMBER_OF_TIMES_TO_REPEAT_EACH_OVERRIDE]
```
