#! encoding: utf-8

import os
import random
import numpy as np
import sys

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.
    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, data_dir, pairs_filepath, img_ext):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext
        if os.path.exists(self.pairs_filepath):
            os.remove(self.pairs_filepath)


    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()
        self._shuffle()


    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in os.listdir(self.data_dir):
            if name == ".DS_Store" or ".txt" in name:
                continue

            a = []
            for file in os.listdir(os.path.join(self.data_dir, name)):
                if file == ".DS_Store" or ".txt" in file:
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                for i in range(len(a)-1):
                   file1 = random.choice(a)
                   file2 = random.choice(a)
                   if file1 == file2:
                       continue
                   filepath1 = os.path.join(self.data_dir, name, file1)
                   filepath2 = os.path.join(self.data_dir, name, file2)
                   f.write(f"{filepath1}\t{filepath2}\ttrue\n") 


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """

        remaining = os.listdir(self.data_dir)
        for i, name in enumerate(os.listdir(self.data_dir)):
           for file in os.listdir(os.path.join(self.data_dir, name)):
               filepath = os.path.join(self.data_dir, name, file)
               other_dir = random.choice(remaining)
               if other_dir == name:
                   continue
               other_file = os.path.join(self.data_dir, other_dir)
               other_file = random.choice(os.listdir(other_file))
               other_file = os.path.join(self.data_dir, other_dir, other_file)
               with open(self.pairs_filepath, "a") as f:
                   f.write(f"{filepath}\t{other_file}\tfalse\n") 

    def _shuffle(self):
        with open(self.pairs_filepath, "r") as f:
            lest = []
            for line in f.readlines():
                lest.append(line)

            lest = np.array(lest, dtype=object)

        with open(self.pairs_filepath, "w") as f:
            f.write("Generated pairs:\n")

        with open(self.pairs_filepath, "a") as f:
            rng = np.random.default_rng()
            for i in range(50):
                rng.shuffle(lest)

            for x in lest:
                f.write(x)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    pairs_filepath = sys.argv[2]
    img_ext = ".jpg"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext)
    generatePairs.generate()
