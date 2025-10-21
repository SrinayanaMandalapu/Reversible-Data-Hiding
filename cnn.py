# cnn.py
import os
import cv2
import time
import math
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
import random
from tqdm import tqdm
import time
from tensorflow.keras import layers, models

# ----------------------- CNN model -----------------------
def build_cnn(input_shape=(9, 9, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # predict intensity
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ----------------------- Create training data -----------------------
# def create_training_data(dataset_path, patch_size=9, max_images=100):
#     print(f"[DEBUG] Checking dataset path: {dataset_path}")
#     print(f"[DEBUG] Found {len(os.listdir(dataset_path))} items in folder")
#     X, y = [], []
#     pad = patch_size // 2
#     count = 0

#     for root, _, files in os.walk(dataset_path):
#         for fname in files:
#             if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
#                 img_path = os.path.join(root, fname)
#                 img = imageio.imread(img_path)
#                 if img.ndim == 3:
#                     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#                 img = img.astype(np.uint8)
#                 img_padded = np.pad(img, pad, mode='reflect')

#                 # collect patches
#                 for i in range(pad, pad + img.shape[0], 4):  # step for efficiency
#                     for j in range(pad, pad + img.shape[1], 4):
#                         patch = img_padded[i-pad:i+pad+1, j-pad:j+pad+1]
#                         X.append(patch)
#                         y.append(img[i-pad, j-pad])
#                 count += 1
#                 if count >= max_images:
#                     return np.array(X), np.array(y)
#     return np.array(X), np.array(y)

def create_training_data(dataset_path, patch_size=9, max_images=None):
    print(f"[INFO] Preparing training data...")
    X, y = [], []
    image_extensions = (".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    valid_images = []

    # recursively walk through all folders
    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith(image_extensions):
                valid_images.append(os.path.join(root, fname))

    print(f"[DEBUG] Total image files found: {len(valid_images)}")

    count = 0
    for img_path in valid_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        h, w = img.shape
        if h < patch_size or w < patch_size:
            continue

        for _ in range(3):
            y0 = random.randint(0, h - patch_size)
            x0 = random.randint(0, w - patch_size)
            patch = img[y0:y0 + patch_size, x0:x0 + patch_size]
            X.append(patch)
            y.append(1)

        count += 1
        if max_images and count >= max_images:
            break

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[INFO] Total valid images processed: {count}")
    print(f"[INFO] Training on {len(X)} samples...")

    if len(X) == 0:
        raise ValueError("No images were loaded. Check dataset path or image format!")

    X = X.reshape(-1, patch_size, patch_size, 1) / 255.0
    return X, y

# ----------------------- Predict using CNN -----------------------
# def cnn_predictor(img, model):
#     patch_size = model.input_shape[1]
#     pad = patch_size // 2
#     img_padded = np.pad(img, pad, mode='reflect')
#     h, w = img.shape
#     pred = np.zeros_like(img, dtype=np.float32)

#     for i in range(h):
#         for j in range(w):
#             patch = img_padded[i:i + patch_size, j:j + patch_size].reshape(1, patch_size, patch_size, 1) / 255.0
#             pred[i, j] = model.predict(patch, verbose=0)[0][0] * 255.0

#     return np.clip(pred, 0, 255).astype(np.uint8)

def cnn_predictor(img, model, patch_size=None, batch_size=256):
    """
    Predicts output for a full grayscale image using a patch-based CNN.

    Parameters:
    - img: 2D numpy array (grayscale)
    - model: trained Keras CNN model
    - patch_size: size of patches (will use model.input_shape if None)
    - batch_size: number of patches to predict at once

    Returns:
    - pred: 2D numpy array same size as input image
    """
    if patch_size is None:
        patch_size = model.input_shape[1]  # 9

    pad = patch_size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    h, w = img.shape
    pred = np.zeros((h, w), dtype=np.float32)

    # Collect all patches
    patches = []
    positions = []

    for i in range(h):
        for j in range(w):
            patch = img_padded[i:i+patch_size, j:j+patch_size].astype(np.float32) / 255.0
            patch = np.expand_dims(patch, axis=-1)  # (patch_size, patch_size, 1)
            patches.append(patch)
            positions.append((i, j))

    patches = np.array(patches)
    
    # Predict in batches
    for start in tqdm(range(0, len(patches), batch_size), desc="CNN prediction"):
        end = start + batch_size
        batch = patches[start:end]
        preds = model.predict(batch, verbose=0)
        for idx, (i, j) in enumerate(positions[start:end]):
            pred[i, j] = preds[idx][0] * 255.0

    return np.clip(pred, 0, 255).astype(np.uint8)

# def cnn_predictor(img, model, target_size=(64, 64), channels=1):
#     """
#     Predicts output from a CNN model safely.
    
#     Parameters:
#     - img: input image as numpy array
#     - model: trained Keras CNN model
#     - target_size: tuple of (height, width) the model expects
#     - channels: number of channels the model expects (1 for grayscale, 3 for RGB)
    
#     Returns:
#     - prediction from the model
#     """
#     try:
#         # Resize image to the model's expected input size
#         img_resized = cv2.resize(img, target_size)

#         # Ensure the correct number of channels
#         if channels == 1 and len(img_resized.shape) == 3:
#             img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
#         elif channels == 3 and len(img_resized.shape) == 2:
#             img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

#         # Normalize pixel values
#         img_resized = img_resized / 255.0

#         # Add batch dimension and channel dimension if needed
#         if channels == 1:
#             img_resized = np.expand_dims(img_resized, axis=-1)  # (h, w, 1)
#         img_resized = np.expand_dims(img_resized, axis=0)       # (1, h, w, c)

#         # Make prediction
#         prediction = model.predict(img_resized)
#         return prediction

#     except Exception as e:
#         print("[CNN ERROR] during prediction:", e)
#         return None

# ----------------------- Train and save model -----------------------
def train_cnn_model(dataset_path, model_path="cnn_model.h5"):
    print(f"[INFO] Preparing training data...")
    print(f"[DEBUG] Checking dataset path: {dataset_path}")
    print(f"[DEBUG] Found {len(os.listdir(dataset_path))} items in folder")
    X, y = create_training_data(dataset_path, patch_size=9, max_images=50)
    X = X.reshape(-1, 9, 9, 1) / 255.0
    y = y / 255.0

    print(f"[INFO] Training on {len(X)} samples...")
    model = build_cnn()
    model.fit(X, y, epochs=3, batch_size=64, verbose=1)
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")
    return model

# ----------------------- Demo per image -----------------------
def demo_cnn(image_path, model):
    t0 = time.perf_counter()
    img = imageio.imread(image_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)

    pred = cnn_predictor(img, model)
    diff = img.astype(np.float32) - pred.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    psnr = float('inf') if mse == 0 else 10 * math.log10((255.0 * 255.0) / mse)
    total_time = time.perf_counter() - t0

    return {
        "height": img.shape[0],
        "width": img.shape[1],
        "mse": mse,
        "psnr_db": psnr,
        "time_total_s": total_time
    }
