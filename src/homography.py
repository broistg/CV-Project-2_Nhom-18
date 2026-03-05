import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_homography(src_pts, dst_pts, method=cv.RANSAC):
    H, mask = cv.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), method, 5.0)
    return H, mask

def apply_transform(image, H, size=None):
    h, w = image.shape[:2]
    return cv.warpPerspective(image, H, size if size else (w, h))

def overlay_image(background, foreground, H):

    h_fg, w_fg = foreground.shape[:2]
    h_bg, w_bg = background.shape[:2]

    warped_fg = cv.warpPerspective(foreground, H, (w_bg, h_bg))

    mask = np.ones((h_fg, w_fg), dtype=np.uint8) * 255
    warped_mask = cv.warpPerspective(mask, H, (w_bg, h_bg))

    result = background.copy()
    result[warped_mask > 0] = warped_fg[warped_mask > 0]

    return result

def automatic_find_dst_pts(template_img, background_img):
    orb = cv.ORB_create(nfeatures=2000)
    
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(background_img, None)
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    h, w = template_img.shape[:2]
    template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    found_corners = cv.perspectiveTransform(template_corners, H)
    
    return found_corners.reshape(4, 2), H