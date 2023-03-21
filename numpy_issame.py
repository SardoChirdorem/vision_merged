import numpy as np
import os
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

def mkarr(input_file, output_file):
    empty_list = []
    with open(input_file) as f:
        for line in f.readlines():
            truth_value = True if line.strip().split()[-1] == "true" else False
            temp_arr = np.array([truth_value])
            empty_list.append(temp_arr)
        
    np.save(output_file, np.concatenate(empty_list))


if __name__ == "__main__":
    mkarr(input_file, output_file)