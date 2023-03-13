import bcolz
import numpy as np
import cv2 as cv
import os
import sys

def looper(path):
    empty_array = []

    for i, img in enumerate(os.listdir(path)):
        im = cv.imread(os.path.join(path, img))
        empty_array.append(im)

    return empty_array


def flusher(root, destination):
    empty_array = []
    for i, dir in enumerate(os.listdir(root)):
        path = os.path.join(root, dir)
        arr = looper(path)
        empty_array = empty_array + arr

    carray = bcolz.carray(empty_array, rootdir=destination)
    carray.flush()



if __name__ == "__main__":
    root = sys.argv[1]
    destination = sys.argv[2]
    flusher(root, destination)