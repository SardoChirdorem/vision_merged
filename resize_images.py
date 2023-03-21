import cv2
import os
import shutil
from tqdm import tqdm
from uuid import uuid4
import sys

SIZE = 160

class Resize_Images:
    def __init__(self, PATH_input_dir, PATH_output_dir) -> None:
        self.PATH_input_dir = PATH_input_dir
        self.PATH_output_dir = PATH_output_dir

        # os.makedirs(PATH_output_dir, exist_ok=True)
        try:
            shutil.copytree(self.PATH_input_dir, self.PATH_output_dir, ignore=self.ignore_files)
        except:
            pass

    def ignore_files(self, dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]


    def resize(self):    
        for folder in tqdm(os.listdir(self.PATH_input_dir)):
            PATH_folder = os.path.join(self.PATH_input_dir, folder)
            for image in os.listdir(PATH_folder):
                img = cv2.imread(os.path.join(PATH_folder, image))
                img = cv2.resize(img, (SIZE, SIZE))
                cv2.imwrite(os.path.join(self.PATH_output_dir, folder, image.replace(' ', '_')), img)

if __name__ == "__main__":
    PATH_input_folder = sys.argv[1]
    PATH_output = sys.argv[2]
    obj = Resize_Images(PATH_input_folder, PATH_output)
    obj.resize()
