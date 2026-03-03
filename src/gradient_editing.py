import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh

background = cv2.imread("../data/inputs/gde/background.jpg")
source = cv2.imread("../data/inputs/gde/source.jpg")

if background is None or source is None:
    print("Không tìm thấy ảnh!")
    exit()

background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB) #Đổi dạng bgr sang rgb, dạng rgb chỉ dùng cho việc hiển thị ảnh bằng matplotlib, còn openCV phải xử lý ảnh dạng bgr
source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

bg_h, bg_w = background.shape[:2] #Nếu lấy dầy đủ [:] sẽ có dạng (height, width, channels)
src_h, src_w = source.shape[:2]

# 2. Tạo mask hình tròn

mask = np.zeros((src_h, src_w), dtype=np.uint8)

center_mask = (src_w // 2, src_h // 2)
radius = min(src_w, src_h) // 3

cv2.circle(mask, center_mask, radius, 255, -1)

# 3. Tính offset an toàn

x_offset = 200
y_offset = 150

# Đảm bảo không tràn biên nếu kích thước offset vượt biên sẽ lấy offset tối đa có thể 
if x_offset + src_w > bg_w:
    x_offset = bg_w - src_w

if y_offset + src_h > bg_h:
    y_offset = bg_h - src_h

# 4. Ghép trực tiếp (Copy-Paste an toàn)

direct_clone = background.copy()

for i in range(src_h):
    for j in range(src_w):
        if mask[i, j] == 255:
            direct_clone[y_offset + i, x_offset + j] = source[i, j]

direct_clone_rgb = cv2.cvtColor(direct_clone, cv2.COLOR_BGR2RGB)

# 5. Poisson Blending (An toàn)

center = (
    x_offset + src_w // 2,
    y_offset + src_h // 2
)

poisson_result = cv2.seamlessClone(
    source,
    background,
    mask,
    center,
    cv2.NORMAL_CLONE
)

poisson_rgb = cv2.cvtColor(poisson_result, cv2.COLOR_BGR2RGB)

# 6. Hiển thị

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