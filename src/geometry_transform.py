import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image():
    base_dir = os.path.dirname(__file__)
    img_path = os.path.join(base_dir, "../data/inputs/geometry/bg1.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image not found")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def translate(image, tx, ty):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))


def rotate(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, M, (new_w, new_h))


def scale(image, fx, fy):
    return cv2.resize(image, None, fx=fx, fy=fy)


def affine_transform(image, pts1, pts2):
    h, w = image.shape[:2]
    M = cv2.getAffineTransform(np.float32(pts1), np.float32(pts2))

    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, M)

    x_min, y_min = np.int32(transformed.min(axis=0).ravel())
    x_max, y_max = np.int32(transformed.max(axis=0).ravel())

    M[0, 2] += -x_min
    M[1, 2] += -y_min

    new_w = x_max - x_min
    new_h = y_max - y_min

    return cv2.warpAffine(image, M, (new_w, new_h))


def save_outputs(images_dict):
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "../data/outputs")
    os.makedirs(output_dir, exist_ok=True)

    for name, img in images_dict.items():
        path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def display_results(images_dict):
    plt.figure(figsize=(15, 10))
    for i, (title, image) in enumerate(images_dict.items()):
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    img = load_image()

    translated = translate(img, 80, 40)
    rotated = rotate(img, 30)
    scaled_up = scale(img, 1.5, 1.5)
    scaled_down = scale(img, 0.5, 0.5)

    pts1 = [[50, 50], [200, 50], [50, 200]]
    pts2 = [[70, 80], [220, 60], [80, 250]]
    affine_img = affine_transform(img, pts1, pts2)

    results = {
        "Original": img,
        "Translation": translated,
        "Rotation": rotated,
        "Scale_Up": scaled_up,
        "Scale_Down": scaled_down,
        "Affine": affine_img
    }

    save_outputs(results)
    display_results(results)


if __name__ == "__main__":
    main()