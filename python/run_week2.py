import os
import cv2
from pathlib import Path
from week2.filters import medianBlur, median_blur_accelerated
from week2.ransac import ransacMatching, find_features, match_features
import numpy as np
import time

input_dir = 'data/week2'
output_dir = 'data/week2'


def test_median_blur_accelerated(image):
    image_blurred = median_blur_accelerated(image, kernel=np.zeros(shape=(4, 5)))

    output_file_name = '{}/{}_{}'.format(output_dir, 'bw_accelerated_blurred', file_name)

    print(output_file_name)

    cv2.imwrite(str(Path(output_file_name).resolve()), image_blurred)


def test_median_blur(image):
    image_blurred = medianBlur(image, kernel=np.zeros(shape=(4, 5)))

    output_file_name = '{}/{}_{}'.format(output_dir, 'bw_naive_blurred', file_name)

    print(output_file_name)

    cv2.imwrite(str(Path(output_file_name).resolve()), image_blurred)


def test_ransac(image):
    # Get feature points
    from week1.augmentation.utils import image_rotation

    rows, cols = image.shape[:2]

    rotated_image = image_rotation(image)

    kp1, desc1 = find_features(image)
    kp2, desc2 = find_features(rotated_image)

    A, B = match_features(kp1, kp2, desc1, desc2)

    homography = ransacMatching(A[:len(B)], B)

    dist = cv2.warpPerspective(image, homography, (cols, rows))

    # cv2.imshow('original.jpg', image)
    # cv2.imshow('rotated.jpg', image2)
    # cv2.imshow('ransacmatch.jpg', dist)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()

    output_file_name = '{}/{}_{}'.format(output_dir, 'rotated', file_name)

    print(output_file_name)

    cv2.imwrite(str(Path(output_file_name).resolve()), rotated_image)

    output_file_name = '{}/{}_{}'.format(output_dir, 'ransac', file_name)

    print(output_file_name)

    cv2.imwrite(str(Path(output_file_name).resolve()), dist)


def test_sift(image):
    img = image
    # create sift class
    sift = cv2.xfeatures2d.SIFT_create()
    # detect SIFT
    kp = sift.detect(img, None)  # None for mask
    # compute SIFT descriptor
    kp, desc = sift.compute(img, kp)

    print(desc.shape)
    img_sift = cv2.drawKeypoints(img, kp, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output_file_name = '{}/{}_{}'.format(output_dir, 'sift', file_name)

    print(output_file_name)

    cv2.imwrite(str(Path(output_file_name).resolve()), img_sift)


if __name__ == '__main__':

    for file_name in os.listdir(input_dir):
        input_file_name = Path(input_dir + '/' + file_name).resolve()
        print(file_name)
        image = cv2.imread(str(input_file_name), 0)

        output_file_name = '{}/{}_{}'.format(output_dir, 'bw_unblurred', file_name)
        cv2.imwrite(str(Path(output_file_name).resolve()), image)

        start = time.time()

        test_median_blur(image)

        end = time.time()

        print('Native algorithm: running time--- {}'.format(end - start))

        start = time.time()

        test_median_blur_accelerated(image)

        end = time.time()

        print('Numpy broadcast: running time--- {}'.format(end - start))

        test_sift(image)

        test_ransac(image)


