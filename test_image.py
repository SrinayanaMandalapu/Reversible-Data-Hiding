# import os
# from utils import classify_image  # if it's saved as utils.py

# # If not inside utils.py, paste your function definitions above this line.

# image_path = os.path.join("BOSSbase_1.01", "1000.pgm")

# label, score = classify_image(image_path, threshold=1500)

# print(f"Image: {image_path}")
# print(f"Complexity Score: {score:.2f}")
# print(f"Label: {label.upper()}")

from cnn import cnn_predictor
from tensorflow.keras.models import load_model
import imageio.v2 as imageio
import cv2

model = load_model("cnn_model.h5", compile=False)
img = imageio.imread("BOSSbase_1.01/1002.pgm")
if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
pred = cnn_predictor(img, model)
print("Prediction successful, shape:", pred.shape)
