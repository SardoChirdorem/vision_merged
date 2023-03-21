import bcolz
import numpy as np
import cv2 as cv
import os
import sys
from tqdm import tqdm
from resize_images import Resize_Images

def looper(path):
    empty_array = np.empty([1,160,160,3], dtype="uint8")

    for i, img in enumerate(os.listdir(path)):
        im = cv.imread(os.path.join(path, img))
        new_shape = (1,) + im.shape
        im = im.reshape(new_shape)
        empty_array = np.append(empty_array, im, axis=0)

    return empty_array


def flusher(root, destination):
    resize_dir = os.path.join(os.path.split(root)[0], os.path.split(root)[1] + "_resized")
    resizer = Resize_Images(root, resize_dir)
    resizer.resize()
    root = resize_dir

    empty_array = []
    for i, dir in tqdm(enumerate(os.listdir(root)), total=len(os.listdir(root))):
        path = os.path.join(root, dir)
        arr = looper(path)
        empty_array.append(arr)

    flush = np.concatenate(empty_array, axis=0)

    carray = bcolz.carray(flush, rootdir=destination)
    carray.flush()



if __name__ == "__main__":
    root = sys.argv[1]
    destination = sys.argv[2]
    flusher(root, destination)