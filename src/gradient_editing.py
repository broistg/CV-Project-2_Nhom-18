import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images(bg_path, src_path):
    background = cv2.imread(bg_path)
    source = cv2.imread(src_path)

    if background is None or source is None:
        raise ValueError("Không tìm thấy ảnh!")

    return background, source


def create_circle_mask(src_h, src_w):
    mask = np.zeros((src_h, src_w), dtype=np.uint8)

    center_mask = (src_w // 2, src_h // 2)
    radius = min(src_w, src_h) // 3

    cv2.circle(mask, center_mask, radius, 255, -1)

    return mask


def compute_safe_offset(bg_w, bg_h, src_w, src_h, x_offset, y_offset):

    if x_offset + src_w > bg_w:
        x_offset = bg_w - src_w

    if y_offset + src_h > bg_h:
        y_offset = bg_h - src_h

    return x_offset, y_offset


def direct_paste(background, source, mask, x_offset, y_offset):

    src_h, src_w = source.shape[:2]
    result = background.copy()

    for i in range(src_h):
        for j in range(src_w):
            if mask[i, j] == 255:
                result[y_offset + i, x_offset + j] = source[i, j]

    return result


def poisson_blend(background, source, mask, x_offset, y_offset):

    src_h, src_w = source.shape[:2]

    center = (
        x_offset + src_w // 2,
        y_offset + src_h // 2
    )

    result = cv2.seamlessClone(
        source,
        background,
        mask,
        center,
        cv2.NORMAL_CLONE
    )

    return result


def show_results(source, direct_clone, poisson_result, mask):

    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    direct_clone_rgb = cv2.cvtColor(direct_clone, cv2.COLOR_BGR2RGB)
    poisson_rgb = cv2.cvtColor(poisson_result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15,5))

    plt.subplot(1,4,1)
    plt.title("Source")
    plt.imshow(source_rgb)
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Direct Paste")
    plt.imshow(direct_clone_rgb)
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("Poisson Blending")
    plt.imshow(poisson_rgb)
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()