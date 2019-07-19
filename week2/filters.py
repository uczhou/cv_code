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
    return np.lib.stride_tricks.as_strided(image,
                                           shape=[rows - kernel_rows + 1, cols - kernel_cols + 1, kernel_rows, kernel_cols],
                                           strides=image.strides + image.strides)


def find_median(vector):
    m, n = vector.shape
    vector_copy = np.copy(vector).reshape(1, m*n)

    # np.ndarray.sort(vector_copy)

    size = m * n

    if size % 2 == 1:
        return quick_select(vector_copy[0, :], size // 2)
    else:
        return (quick_select(vector_copy[0, :], size // 2 - 1) + quick_select(vector_copy[0, :], size // 2)) / 2

    # return (sum(vector_copy[0, size // 2 - 1:size // 2 + 1]) / 2.0, vector_copy[0, size // 2])[size % 2]


def quick_select(vector, kth):
    # O(n)

    def select(vector, start, end, kth):

        if start == end:
            return vector[start]

        pivot = np.random.randint(start, end)

        vector[start], vector[pivot] = vector[pivot], vector[start]

        vl, vr = start + 1, end
        while vl <= vr:
            if vector[vl] > vector[start] and vector[vr] <= vector[start]:
                vector[vl], vector[vr] = vector[vr], vector[vl]
                vl += 1
                vr -= 1
            elif vector[vl] <= vector[start]:
                vl += 1
            else:
                vr -= 1

        vector[start], vector[vl] = vector[vl], vector[start]

        if vl == kth:
            return vector[vl]

        elif vl < kth:
            return select(vector, vl+1, end, kth)

        else:
            return select(vector, start, vl - 1, kth)

    if vector is None or kth < 0 or kth >= len(vector):
        return None


    return select(vector, 0, len(vector)-1, kth)


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
        img_padded = np.lib.pad(img, [(padding_rows_up, padding_rows_down), (padding_cols_left, padding_cols_right)],
                                mode=padding_mode, constant_values=0)
    else:
        img_padded = np.lib.pad(img, [(padding_rows_up, padding_rows_down), (padding_cols_left, padding_cols_right)],
                                mode=padding_mode)

    dist = np.zeros(shape=(rows, cols))

    for i in range(rows):
        for j in range(cols):
            # dist[i][j] = np.median(img_padded[i:i+kernel_rows, j: j+kernel_cols]).astype('uint8')
            dist[i][j] = find_median(img_padded[i:i+kernel_rows, j: j+kernel_cols]).astype('uint8')
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

    dist = np.zeros(shape=(rows, cols))

    for i in range(rows):
        # Calculate row by to avoid memory segmentation fault if image is too large.
        strided = strided_app(img_padded[i: i + kernel_rows, :], kernel)

        dist[i, :] = np.median(strided, axis=(2, 3)).astype('uint8')

    return dist
