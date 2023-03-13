import os
from PIL import Image
import sys

base_path = sys.argv[1]
def converter(path):
    for im in os.listdir(path):
        im_path = os.path.join(path, im)
        im_name = im.split(".")[0]
        img = Image.open(im_path)
        img_p = img.convert("P")
        if not os.path.exists(os.path.join(os.path.split(path)[0], "data_blp")):
            os.makedirs(os.path.join(os.path.split(path)[0], "data_blp"))
        img_p.save(os.path.join(os.path.split(path)[0], fr"data_blp\{im_name}.blp"))

def data_iterator(base_path):
    for folder in os.listdir(base_path):
        path = os.path.join(base_path, folder)
        converter(path)

if __name__ == "__main__":
    data_iterator(base_path)