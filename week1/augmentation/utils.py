import cv2
import numpy as np
import random


def is_grayscale_image(image):
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[2] == 1)


def image_flip(image, axis=1):
    '''
    :param image: numpy array
    :param axis: flip axis
    :return: numpy array
    '''
    flipped_image = cv2.flip(image, axis)
    return flipped_image


def image_translate(image, axis_x=0, axis_y=0):
    rows, cols = image.shape[:2]

    M = np.float32([1,0, axis_x], [0, 1, axis_y])
    translated_image = cv2.warpAffine(image, M, (cols, rows))

    return translated_image


def image_color_shift(image, upper_bound=50, lower_bound=-50):

    layers = cv2.split(image)

    for layer in layers:
        rand_val = random.randint(lower_bound, upper_bound)
        if rand_val == 0:
            pass
        elif rand_val > 0:
            lim = 255 - rand_val
            layer[layer > lim] = 255
            layer[layer <= lim] = (rand_val + layer[layer <= lim]).astype(image.dtype)
        elif rand_val < 0:
            lim = 0 - rand_val
            layer[layer < lim] = 0
            layer[layer >= lim] = (rand_val + layer[layer >= lim]).astype(image.dtype)

    img_merge = cv2.merge(layers)
    # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge


def image_crop(image, row_start=0, row_end=100, col_start=0, col_end=200):

    return image[row_start:row_end, col_start:col_end]


def image_gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(image, table)


def image_histogram(image, scale=0.5):
    rows, cols = image.shape[:2]
    image_small = cv2.resize(image, (int(cols * scale), int(rows * scale)))

    image_yuv = cv2.cvtColor(image_small, cv2.COLOR_BGR2YUV)

    image_yuv[:, :, :] = cv2.equalizeHist(image_yuv[:, :, :])

    return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)


def image_rotation(image, angle=30, scale=1.0):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((int(cols / 2), int(rows / 2)), angle=angle, scale=scale)
    image_rotate = cv2.warpAffine(image, M, (cols, rows))

    return image_rotate


def image_similarity_transform(image, angle=30, scale=0.5):

    return image_rotation(image, angle=angle, scale=scale)


def image_affine_transform(image, pts1, pts2):
    rows, cols = image.shape[:2]

    np_pts1 = np.float32(pts1)
    np_pts2 = np.float32(pts2)

    M = cv2.getAffineTransform(np_pts1, np_pts2)
    return cv2.warpAffine(image, M, (cols, rows))


def image_perspective_transform(image, pts1, pts2):
    rows, cols = image.shape[:2]
    np_pts1 = np.float32(pts1)
    np_pts2 = np.float32(pts2)

    M = cv2.getPerspectiveTransform(np_pts1, np_pts2)
    return cv2.warpPerspective(image, M, (cols, rows))


def image_adaptive_histogram(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if is_grayscale_image(image):
        image = clahe.apply(image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        image[:, :, 0] = clahe.apply(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image


def image_normalize(image, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    image = image.astype(np.float32)
    image -= mean
    image *= denominator

    return image


def image_blur(image, ksize):

    return cv2.blur(image, ksize)


def image_gaussian_blur(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)
