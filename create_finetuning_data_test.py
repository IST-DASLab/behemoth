import os
import unittest

TEST_DATA_DIR = "test_data/finetuning_data_creation"

if os.path.exists(TEST_DATA_DIR):
    os.system("rm -rf " + TEST_DATA_DIR + "/*")
else:
    os.makedirs(TEST_DATA_DIR)


class TestFTDataGeneration(unittest.TestCase):
    def test_whole_script_allsame(self):
        os.system(
            f"python create_data.py -s 50 -r 5 -o 5 -n 1 --shuffle --output-dir {TEST_DATA_DIR}/simple"
        )
        with open(TEST_DATA_DIR + "/simple/data/full_text_0.txt", "r") as f:
            train_data = f.readlines()[0].strip().split("<|endoftext|>")
        with open(
            TEST_DATA_DIR + "/simple/viscera/relationship_graph_quasitokens.txt", "r"
        ) as f:
            orig_triples = [x.strip().split() for x in f.readlines()]
        os.system(
            f"python create_finetuning_data.py --all-same-remapping --num-ft-overrides 1  --num-repeats-per-override 2 -g {TEST_DATA_DIR}/simple"
        )
        with open(
            TEST_DATA_DIR
            + "/simple/ftremapping_1overrides_2overriderepeats_allsame/viscera/edited_relationship_graph_quasitokens.txt",
            "r",
        ) as f:
            remapped_triplet = [x.strip().split() for x in f.readlines()]
        for subject, relationship, new_object in remapped_triplet:
            found_match = False
            for t in orig_triples:
                if t[0] == subject and t[1] == relationship:
                    found_match = True
                    self.assertTrue(t[2] != new_object)
                    continue
            self.assertTrue(found_match)
        # with open(
        #     TEST_DATA_DIR
        #     + "/simple/ftremapping_1overrides_2overriderepeats_allsame/viscera/edited_relationship_graph_quasitokens.txt",
        #     "r",
        # ) as f:
        #     remapped_triplet = [x.strip().split() for x in f.readlines()]
        # for subject, relationship, new_object in remapped_triplet:
        #     found_match = False
        #     for t in orig_triples:
        #         if t[0] == subject and t[1] == relationship:
        #             found_match = True
        #             self.assertTrue(t[2] != new_object)
        #             continue
        #     self.assertTrue(found_match)
        for prefix in ["with_same_value", "with_same_rel", "reg_data"]:
            with open(
                TEST_DATA_DIR
                + f"/simple/ftremapping_1overrides_2overriderepeats_allsame/viscera/{prefix}_relationship_graph_quasitokens.txt",
                "r",
            ) as f:
                remapped_triplet = [x.strip().split() for x in f.readlines()]
            for subject, relationship, new_object in remapped_triplet:
                found_match = False
                for t in orig_triples:
                    if t[0] == subject and t[1] == relationship:
                        found_match = True
                        self.assertTrue(t[2] == new_object)
                        continue
                self.assertTrue(found_match)
        with open(
            TEST_DATA_DIR
            + "/simple/ftremapping_1overrides_2overriderepeats_allsame/finetuning_data/with_same_value_data.txt",
            "r",
        ) as f:
            new_lines = [x.strip() for x in f.readlines()]
        with open(TEST_DATA_DIR + "/simple/data/full_text_0.txt", "r") as f:
            train_data = f.read()
            self.assertTrue(new_lines[0] in train_data)

    def test_whole_script_notallsame(self):
        os.system(
            f"python create_data.py -s 50 -r 5 -o 5 -n 1 --shuffle --output-dir {TEST_DATA_DIR}/simple"
        )
        with open(TEST_DATA_DIR + "/simple/data/full_text_0.txt", "r") as f:
            train_data = f.readlines()[0].strip().split("<|endoftext|>")
        with open(
            TEST_DATA_DIR + "/simple/viscera/relationship_graph_quasitokens.txt", "r"
        ) as f:
            orig_triples = [x.strip().split() for x in f.readlines()]
        os.system(
            f"python create_finetuning_data.py --num-ft-overrides 2 --num-repeats-per-override=2 -g {TEST_DATA_DIR}/simple"
        )
        with open(
            TEST_DATA_DIR
            + "/simple/ftremapping_2overrides_2overriderepeats/viscera/edited_relationship_graph_quasitokens.txt",
            "r",
        ) as f:
            remapped_triplet = [x.strip().split() for x in f.readlines()]
            all_objects = set()
        for subject, relationship, new_object in remapped_triplet:
            all_objects.add(new_object)
            found_match = False
            for t in orig_triples:
                if t[0] == subject and t[1] == relationship:
                    found_match = True
                    self.assertTrue(t[2] != new_object)
                    continue
            self.assertTrue(found_match)
        self.assertTrue(
            len(all_objects) * 2 == len(remapped_triplet),
            "The number of new objects doesn't match the number of remapped rows",
        )
        for prefix in ["with_same_value", "with_same_rel", "reg_data"]:
            with open(
                TEST_DATA_DIR
                + f"/simple/ftremapping_2overrides_2overriderepeats/viscera/{prefix}_relationship_graph_quasitokens.txt",
                "r",
            ) as f:
                remapped_triplet = [x.strip().split() for x in f.readlines()]
            for subject, relationship, new_object in remapped_triplet:
                found_match = False
                for t in orig_triples:
                    if t[0] == subject and t[1] == relationship:
                        found_match = True
                        self.assertTrue(t[2] == new_object)
                        continue
                self.assertTrue(found_match)
        with open(
            TEST_DATA_DIR
            + "/simple/ftremapping_2overrides_2overriderepeats/finetuning_data/with_same_value_data.txt",
            "r",
        ) as f:
            new_lines = [x.strip() for x in f.readlines()]
        with open(TEST_DATA_DIR + "/simple/data/full_text_0.txt", "r") as f:
            train_data = f.read()
            self.assertTrue(new_lines[0] in train_data)


class TestNested_FTDataGeneration(unittest.TestCase):

    def test_whole_script(self):
        os.system(
            f"python create_data.py -s 500 -r 5 -o 10 -n 1 --shuffle --num-nested-objects 5 --output-dir {TEST_DATA_DIR}/nested"
        )
        with open(TEST_DATA_DIR + "/nested/data/full_text_0.txt", "r") as f:
            train_data = f.readlines()[0].strip().split("<|endoftext|>")
        with open(
            TEST_DATA_DIR + "/nested/viscera/relationship_graph_quasitokens.txt", "r"
        ) as f:
            orig_triples = [x.strip().split() for x in f.readlines()]
        for triplet in orig_triples:
            for l in train_data:
                if triplet[0] in l and triplet[1] in l:
                    self.assertTrue(triplet[2] in l)
        for triplet in orig_triples:
            found = [False, False]
            for td in train_data:
                if triplet[2] in td and triplet[5] in td and triplet[4] in td:
                    found[1] = True
            for td in train_data:
                if triplet[0] in td and triplet[5] in td and triplet[3] in td:
                    found[0] = True
            self.assertTrue(found[0] and found[1])

        # Now some FT data
        os.system(
            f"python create_finetuning_data.py --num-ft-overrides 2 --num-repeats-per-override 2 -g {TEST_DATA_DIR}/nested --all-same-remapping"
        )
        with open(
            TEST_DATA_DIR
            + "/nested/ftremapping_2overrides_2overriderepeats_allsame/viscera/edited_relationship_graph_quasitokens.txt",
            "r",
        ) as f:
            remapped_triplet = [x.strip().split() for x in f.readlines()]
            all_objects = set()
        for subject, relationship, new_object in remapped_triplet:
            all_objects.add(new_object)
            found_match = False
            for t in orig_triples:
                if t[0] == subject and t[1] == relationship:
                    found_match = True
                    self.assertTrue(t[2] != new_object)
                    continue
            self.assertTrue(found_match)
        self.assertTrue(
            len(all_objects) == 1, "All rows should be remapped to the same new object"
        )
        for prefix in ["with_same_value", "with_same_rel", "reg_data"]:
            with open(
                TEST_DATA_DIR
                + f"/nested/ftremapping_2overrides_2overriderepeats_allsame/viscera/{prefix}_relationship_graph_quasitokens.txt",
                "r",
            ) as f:
                non_remapped_triplet = [x.strip().split() for x in f.readlines()]
            for subject, relationship, object in non_remapped_triplet:
                found_match = False
                for t in orig_triples:
                    if t[0] == subject and t[1] == relationship:
                        found_match = True
                        self.assertTrue(t[2] == object)
                        continue
                self.assertTrue(found_match)
        with open(
            TEST_DATA_DIR
            + "/nested/ftremapping_2overrides_2overriderepeats_allsame/finetuning_data/with_same_value_data.txt",
            "r",
        ) as f:
            new_lines = [x.strip() for x in f.readlines()]
        with open(TEST_DATA_DIR + "/nested/data/full_text_0.txt", "r") as f:
            train_data = f.read()
            for nl in new_lines:
                self.assertTrue(nl in train_data)
        for fname in (
            "with_same_value",
            "with_same_rel",
            "reg_data",
            "same_subj_nested_obj_edges",
            "same_obj_nested_obj_edges",
        ):
            with open(
                TEST_DATA_DIR
                + f"/nested/ftremapping_2overrides_2overriderepeats_allsame/ft_validations/{fname}_sro.txt",
                "r",
            ) as f:
                valid_data = [x.replace("\t", "") for x in f.readlines()]
                for nl in valid_data:
                    self.assertTrue(nl[:-1] in train_data, f"{nl} not in {fname}")


if __name__ == "__main__":
    unittest.main()
