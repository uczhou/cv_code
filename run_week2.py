import os
import cv2
from pathlib import Path
from week2.filters import medianBlur, median_blur_accelerated
import numpy as np
import time


if __name__ == '__main__':

    input_dir = 'data/week2'
    output_dir = 'data/week2'
    for file_name in os.listdir(input_dir):
        input_file_name = Path(input_dir + '/' + file_name).resolve()
        print(file_name)
        image = cv2.imread(str(input_file_name), 0)

        output_file_name = '{}/{}_{}'.format(output_dir, 'bw_unblurred', file_name)
        cv2.imwrite(str(Path(output_file_name).resolve()), image)

        start = time.time()

        image_blurred = medianBlur(image, kernel=np.zeros(shape=(4,5)))

        output_file_name = '{}/{}_{}'.format(output_dir, 'bw_naive_blurred', file_name)

        print(output_file_name)

        cv2.imwrite(str(Path(output_file_name).resolve()), image_blurred)

        end = time.time()

        print(end - start)

        start = time.time()

        image_blurred = median_blur_accelerated(image, kernel=np.zeros(shape=(4, 5)))

        output_file_name = '{}/{}_{}'.format(output_dir, 'bw_accelerated_blurred', file_name)

        print(output_file_name)

        cv2.imwrite(str(Path(output_file_name).resolve()), image_blurred)

        end = time.time()

        print(end - start)



