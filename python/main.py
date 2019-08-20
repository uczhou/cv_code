import os
import cv2
from pathlib import Path
from week1.augmentation import *

functions = {
    'flip': image_flip,
    'translate': image_translate,
    'color': image_color_shift,
    'crop': image_crop,
    'gamma': image_gamma_correction,
    'histogram': image_histogram,
    'rotation': image_rotation,
    'similarity': image_similarity_transform,
    'affine': image_affine_transform,
    'perspective': image_perspective_transform,
    'adaptive': image_adaptive_histogram,
    'blur': image_blur,
    'gaussian_blur': image_gaussian_blur
}

if __name__ == '__main__':

    input_dir = 'data/input'
    output_dir = 'data/output'
    for file_name in os.listdir(input_dir):
        input_file_name = Path(input_dir + '/' + file_name).resolve()
        print(file_name)
        image = cv2.imread(str(input_file_name))

        for key, value in functions.items():
            dist = value(image)

            output_file_name = '{}/{}_{}'.format(output_dir, key, file_name)
            print(output_file_name)

            cv2.imwrite(str(Path(output_file_name).resolve()), dist)
