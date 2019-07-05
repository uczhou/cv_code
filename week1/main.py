import os
import cv2
from pathlib import Path

if __name__ == '__main__':

    input_dir = 'data/input'
    output_dir = 'data/output'
    for file_name in os.listdir(input_dir):
        input_file_name = Path(input_dir + '/' + file_name).resolve()
        print(file_name)
        image = cv2.imread(str(input_file_name))

        dist = image_flip(image, axis=0)

        output_file_name = output_dir + '/' + file_name
        cv2.imwrite(str(Path(output_file_name).resolve()), dist)
