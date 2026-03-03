import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# LOAD IMAGE (SAFE PATH)
# ==============================

base_dir = os.path.dirname(__file__)
img_path = os.path.join(base_dir, "../data/inputs/geometry/bg1.jpg")

img = cv2.imread(img_path)

if img is None:
    print("Không tìm thấy ảnh!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img.shape[:2]

# ==============================
# TRANSLATION
# ==============================
tx, ty = 80, 40
M_trans = np.float32([[1, 0, tx],
                      [0, 1, ty]])

translated = cv2.warpAffine(img, M_trans, (w, h))

# ==============================
# ROTATION 
# ==============================
angle = 30
center = (w // 2, h // 2)

M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

cos = np.abs(M_rot[0, 0])
sin = np.abs(M_rot[0, 1])

new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

M_rot[0, 2] += (new_w / 2) - center[0]
M_rot[1, 2] += (new_h / 2) - center[1]

rotated = cv2.warpAffine(img, M_rot, (new_w, new_h))

# ==============================
# SCALING
# ==============================
scale_up = 1.5
scale_down = 0.5

scaled_up = cv2.resize(img, None, fx=scale_up, fy=scale_up)
scaled_down = cv2.resize(img, None, fx=scale_down, fy=scale_down)

# ==============================
# AFFINE 
# ==============================

pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])

pts2 = np.float32([[70, 80],
                   [220, 60],
                   [80, 250]])

M_affine = cv2.getAffineTransform(pts1, pts2)

corners = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
]).reshape(-1, 1, 2)

transformed = cv2.transform(corners, M_affine)

x_min, y_min = np.int32(transformed.min(axis=0).ravel())
x_max, y_max = np.int32(transformed.max(axis=0).ravel())

M_affine[0, 2] += -x_min
M_affine[1, 2] += -y_min

new_w = x_max - x_min
new_h = y_max - y_min

affine = cv2.warpAffine(img, M_affine, (new_w, new_h))

# ==============================
# SAVE OUTPUT
# ==============================

output_dir = os.path.join(base_dir, "../data/outputs")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cv2.imwrite(os.path.join(output_dir, "output_translation.jpg"), cv2.cvtColor(translated, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "output_rotation.jpg"), cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "output_scale_up.jpg"), cv2.cvtColor(scaled_up, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "output_scale_down.jpg"), cv2.cvtColor(scaled_down, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "output_affine.jpg"), cv2.cvtColor(affine, cv2.COLOR_RGB2BGR))

# ==============================
# DISPLAY
# ==============================

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(translated)
plt.title("Translation")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(rotated)
plt.title("Rotation")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(scaled_up)
plt.title("Scale Up")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(scaled_down)
plt.title("Scale Down")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(affine)
plt.title("Affine")
plt.axis("off")

plt.tight_layout()
plt.show()