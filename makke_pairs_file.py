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
            for file in os.listdir(self.data_dir + "\\" + name):
                if file == ".DS_Store" or ".txt" in file:
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                for i in range(len(a)-1):
                    # temp = random.choice(a).split("0") # This line may vary depending on how your images are named.
                    # w = temp[0].rstrip("_")
                    l = random.choice(a)
                    r = random.choice(a)
                    if l == r:
                        continue
                    f.write(os.path.join(name, l) + "\t" + os.path.join(name, r) + "\n")


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if name == ".DS_Store" or ".txt" in name:
                continue

            remaining = os.listdir(self.data_dir)
            remaining = [f_n for f_n in remaining]
            # del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)
            with open(self.pairs_filepath, "a") as f:
                for i in range(len(os.listdir(self.data_dir + "\\" + name))):
                    ref_list = []
                    other_dir = random.choice(remaining)
                    if ".txt" in other_dir or other_dir == name:
                        continue
                    file1 = random.choice(os.listdir(self.data_dir + "\\" + name))
                    file2 = random.choice(os.listdir(self.data_dir + "\\" + other_dir))
                    str_w = (os.path.join(name, file1) + "\t" + os.path.join(other_dir, file2) + "\tf\n")
                    if str_w in ref_list:
                        continue
                    f.write(str_w)
                    ref_list.append(str_w)

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
