import cv2
import numpy as np


def translation(img, tx, ty):
    h, w = img.shape[:2]

    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    return cv2.warpAffine(img, M, (w, h))


def rotation(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, M, (new_w, new_h))


def scaling(img, scale):
    return cv2.resize(img, None, fx=scale, fy=scale)


def affine_transform(img, pts1, pts2):
    h, w = img.shape[:2]

    M = cv2.getAffineTransform(pts1, pts2)

    corners = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 1, 2)

    transformed = cv2.transform(corners, M)

    x_min, y_min = np.int32(transformed.min(axis=0).ravel())
    x_max, y_max = np.int32(transformed.max(axis=0).ravel())

    M[0, 2] += -x_min
    M[1, 2] += -y_min

    new_w = x_max - x_min
    new_h = y_max - y_min

    return cv2.warpAffine(img, M, (new_w, new_h))