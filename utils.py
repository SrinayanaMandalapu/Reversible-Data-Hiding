import cv2
import numpy as np
import imageio.v2 as imageio

def compute_complexity(image_path):
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)
    return cv2.Laplacian(img, cv2.CV_64F).var()

def classify_image(image_path, threshold=1500):
    score = compute_complexity(image_path)
    label = "complex" if score > threshold else "normal"
    return label, score
