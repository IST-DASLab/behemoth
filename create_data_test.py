import os
import unittest

TEST_DATA_DIR = "test_data/data_creation"

if os.path.exists(TEST_DATA_DIR):
    os.system("rm -rf " + TEST_DATA_DIR + "/*")
else:
    os.makedirs(TEST_DATA_DIR)


class TestDataGeneration(unittest.TestCase):
    def test_whole_script(self):
        os.system(
            f"python create_data.py -s 10 -r 5 -o 5 -n 1 --shuffle --output-dir {TEST_DATA_DIR}/simple"
        )
        with open(TEST_DATA_DIR + "/simple/data/full_text_0.txt", "r") as f:
            train_data = f.readlines()[0].strip().split("<|endoftext|>")
        with open(
            TEST_DATA_DIR + "/simple/metadata/relationship_graph_tokenized.txt", "r"
        ) as f:
            triples = [x.strip().split() for x in f.readlines()]
        for triplet in triples:
            for l in train_data:
                if triplet[0] in l and triplet[1] in l:
                    self.assertTrue(triplet[2] in l)


class TestNestedDataGeneration(unittest.TestCase):

    def test_whole_script(self):
        os.system(
            f"python create_data.py -s 10 -r 5 -o 10 -n 1 --shuffle --num-nested-objects 5 --output-dir {TEST_DATA_DIR}/nested"
        )
        with open(TEST_DATA_DIR + "/nested/data/full_text_0.txt", "r") as f:
            train_data = f.readlines()[0].strip().split("<|endoftext|>")
        with open(
            TEST_DATA_DIR + "/nested/metadata/relationship_graph_tokenized.txt", "r"
        ) as f:
            triples = [x.strip().split() for x in f.readlines()]
        for triplet in triples:
            for l in train_data:
                if triplet[0] in l and triplet[1] in l:
                    self.assertTrue(triplet[2] in l)
        for triplet in triples:
            found = [False, False]
            for td in train_data:
                if triplet[2] in td and triplet[5] in td and triplet[4] in td:
                    found[1] = True
            for td in train_data:
                if triplet[0] in td and triplet[5] in td and triplet[3] in td:
                    found[0] = True
            self.assertTrue(found[0] and found[1])


if __name__ == "__main__":
    unittest.main()
