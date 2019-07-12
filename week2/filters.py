#    Finish 2D convolution/filtering by your self.
#    What you are supposed to do can be described as "median blur", which means by using a sliding window
#    on an image, your task is not going to do a normal convolution, but to find the median value within
#    that crop.
#
#    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When
#    "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version
#    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#    Follow up 1: Can it be completed in a shorter time complexity?
#
#    Python version:
#    def medianBlur(img, kernel, padding_way):
#        img & kernel is List of List; padding_way a string
#        Please finish your code under this blank
#

import numpy as np


def strided_app(image, kernel):
    rows, cols = image.shape[:2]
    kernel_rows, kernel_cols = kernel.shape
    return np.lib.stride_tricks.as_strided(image, shape=[rows - kernel_rows + 1, cols - kernel_cols + 1, kernel_rows, kernel_cols],
                                    strides=image.strides + image.strides)


def medianBlur(img, kernel, padding_way='REPLICA'):

    padding_mode = 'constant' if padding_way == 'ZERO' else 'edge'

    rows, cols = img.shape[:2]
    kernel_rows, kernel_cols = kernel.shape

    # Padding matrix
    padding_rows_up = kernel_rows // 2
    padding_rows_down = kernel_rows // 2 if kernel_rows % 2 == 1 else kernel_rows // 2 - 1

    padding_cols_left = kernel_cols // 2
    padding_cols_right = kernel_cols // 2 if kernel_cols % 2 == 1 else kernel_cols // 2 - 1

    if padding_mode == 'constant':
        img_padded = np.lib.pad(img, [(padding_rows_up, padding_rows_down),(padding_cols_left, padding_cols_right)],
                                mode=padding_mode, constant_values=0)
    else:
        img_padded = np.lib.pad(img, [(padding_rows_up, padding_rows_down),(padding_cols_left, padding_cols_right)],
                                mode=padding_mode)

    dist = np.zeros(shape=(rows,cols))

    for i in range(rows):
        for j in range(cols):
            dist[i][j] = np.median(img_padded[i:i+kernel_rows, j: j+kernel_cols]).astype('uint8')

    return dist


def median_blur_accelerated(img, kernel, padding_way='REPLICA'):

    padding_mode = 'constant' if padding_way == 'ZERO' else 'edge'

    rows, cols = img.shape[:2]
    kernel_rows, kernel_cols = kernel.shape

    # Padding matrix
    padding_rows_up = kernel_rows // 2
    padding_rows_down = kernel_rows // 2 if kernel_rows % 2 == 1 else kernel_rows // 2 - 1

    padding_cols_left = kernel_cols // 2
    padding_cols_right = kernel_cols // 2 if kernel_cols % 2 == 1 else kernel_cols // 2 - 1

    if padding_mode == 'constant':
        img_padded = np.lib.pad(img, [(padding_rows_up, padding_rows_down),(padding_cols_left, padding_cols_right)],
                                mode=padding_mode, constant_values=0)
    else:
        img_padded = np.lib.pad(img, [(padding_rows_up, padding_rows_down),(padding_cols_left, padding_cols_right)],
                                mode=padding_mode)

    dist = np.zeros(shape=(rows,cols))

    for i in range(rows):
        # Calculate row by to avoid memory segmentation fault if image is too large.
        strided = strided_app(img_padded[i: i + kernel_rows, :], kernel)

        dist[i, :] = np.median(strided, axis=(2,3)).astype('uint8')

    return dist
