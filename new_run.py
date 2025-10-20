# hybrid_run.py (fixed for full dataset run)
import os
import csv
import time
import math
import cv2
import numpy as np
import imageio.v2 as imageio
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model

from cnn import train_cnn_model, cnn_predictor
from med import med_predictor
from utils import classify_image


# ------------- Environment + TensorFlow setup ----------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU for stability on Windows


def evaluate(img, pred):
    diff = img.astype(float) - pred.astype(float)
    mse = float((diff ** 2).mean())
    psnr = float('inf') if mse == 0 else 10 * math.log10((255 ** 2) / mse)
    return mse, psnr


def hybrid_process(dataset_path, csv_path="metrics_hybrid.csv"):
    model_path = "cnn_model.h5"

    # Load or train CNN model once
    if os.path.exists(model_path):
        print("[INFO] Loading CNN model...")
        model = load_model(model_path, compile=False)
    else:
        print("[INFO] Training CNN model...")
        model = train_cnn_model(dataset_path, model_path=model_path)

    # Collect all image files
    image_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(dataset_path)
        for fname in files
        if fname.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]

    total_images = len(image_files)
    print(f"[INFO] Found {total_images} valid image files.\n")

    rows = []
    skipped = []

    # Main loop
    for idx, image_path in enumerate(sorted(image_files), start=1):
        rel_path = os.path.relpath(image_path, dataset_path)
        print(f"[INFO] ({idx}/{total_images}) Processing: {rel_path}", flush=True)

        try:
            img = imageio.imread(image_path)
            if img is None or not isinstance(img, np.ndarray):
                print(f"[WARN] Skipping {rel_path}: invalid image data.")
                skipped.append(rel_path)
                continue

            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(np.uint8)

            label, score = classify_image(image_path, threshold=1500)

            t0 = time.perf_counter()
            if label == "complex":
                pred = cnn_predictor(img, model)
            else:
                pred = med_predictor(img)
            total_time = time.perf_counter() - t0

            mse, psnr = evaluate(img, pred)

            rows.append({
                "image_path": rel_path,
                "complexity_score": f"{score:.2f}",
                "label": label,
                "mse": f"{mse:.6f}",
                "psnr_db": "inf" if psnr == float("inf") else f"{psnr:.6f}",
                "time_total_s": f"{total_time:.6f}",
            })

            # Memory cleanup every 50 images
            if idx % 50 == 0:
                gc.collect()
                tf.keras.backend.clear_session()

        except Exception as e:
            print(f"[ERROR] Skipping {rel_path}: {e}")
            skipped.append(rel_path)
            continue

    # Save results
    if rows:
        fieldnames = ["image_path", "complexity_score", "label", "mse", "psnr_db", "time_total_s"]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n✅ [INFO] Saved hybrid metrics to {os.path.abspath(csv_path)}")
        print(f"✅ [INFO] Total images processed successfully: {len(rows)} / {total_images}")

        if skipped:
            with open("skipped_images.txt", 'w') as f:
                for s in skipped:
                    f.write(s + "\n")
            print(f"⚠️ [INFO] Skipped {len(skipped)} images. Details saved in skipped_images.txt")

    else:
        print("[WARNING] No valid images found or processed.")


if __name__ == "__main__":
    dataset_path = "BOSSbase_1.01"
    hybrid_process(dataset_path)
