import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def read_image(path, mode='rgb'):
    """
    Hàm load ảnh an toàn, kiểm tra file tồn tại.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {path}")
    
    if mode == 'gray':
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        # OpenCV đọc BGR, cần chuyển sang RGB
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def normalize_image(image):
    """
    Chuẩn hóa ảnh về dạng hiển thị được (uint8 [0, 255]).
    Thường dùng sau khi thực hiện các phép lọc Sobel/Laplacian (ra float/số âm).
    """
    # Lấy trị tuyệt đối (đối với Sobel/Laplacian biên âm)
    img_abs = np.absolute(image)
    
    # Clip về khoảng 0-255 và chuyển sang uint8
    img_norm = np.clip(img_abs, 0, 255).astype(np.uint8)
    return img_norm

def add_salt_pepper_noise(image, amount=0.05):
    """
    Thêm nhiễu muối tiêu vào ảnh.
    
    Args:
        image: Ảnh đầu vào 2D (Grayscale).
        amount: Tỷ lệ nhiễu (0.0 - 1.0).
    """
    noisy = image.copy()
    
    # Tính số lượng pixel bị nhiễu
    num_salt = int(np.ceil(amount * image.size * 0.5))
    num_pepper = int(np.ceil(amount * image.size * 0.5))
    
    # Tạo tọa độ ngẫu nhiên cho Salt (Trắng - 255)
    coords_salt = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[tuple(coords_salt)] = 255

    # Tạo tọa độ ngẫu nhiên cho Pepper (Đen - 0)
    coords_pepper = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[tuple(coords_pepper)] = 0
    
    return noisy

def show_image(image, title="Image", cmap_type='gray'):
    """
    Hàm hiển thị ảnh đơn bằng Matplotlib.
    """
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')

def show_comparison(img1, img2, title1="Original", title2="Processed", cmap1='gray', cmap2='gray'):
    """
    Hiển thị 2 ảnh song song để so sánh trực quan (Before -> After).
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap=cmap1)
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap2)
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()