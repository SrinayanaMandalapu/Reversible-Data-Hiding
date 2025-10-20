# # test_read_1002.py
# import os
# import imageio.v2 as imageio
# import cv2
# from PIL import Image

# path = os.path.join("BOSSbase_1.01","1002.pgm")
# print("PATH:", path)
# print("Exists:", os.path.exists(path))
# print("Filesize (bytes):", os.path.getsize(path) if os.path.exists(path) else None)

# # Try imageio
# try:
#     im = imageio.imread(path)
#     print("imageio read OK, shape/type:", None if im is None else (im.shape, type(im)))
# except Exception as e:
#     print("imageio error:", e)

# # Try OpenCV grayscale
# try:
#     im_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     print("cv2 read OK, shape/type:", None if im_cv is None else (im_cv.shape, type(im_cv)))
# except Exception as e:
#     print("cv2 error:", e)

# # Try PIL
# try:
#     pil = Image.open(path)
#     print("PIL open OK, mode/size:", pil.mode, pil.size)
#     pil.close()
# except Exception as e:
#     print("PIL error:", e)

# test_inference_1002.py
import os
import imageio.v2 as imageio
import torch
from torchvision import transforms

path = os.path.join("BOSSbase_1.01","1002.pgm")
im = imageio.imread(path)

# Example preprocessing (adjust to your pipeline)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    # add normalization if your model expects it
])

im_tensor = preprocess(im).unsqueeze(0)  # batch dimension
print("Preprocessed tensor shape:", im_tensor.shape)

# Dummy CNN test (replace 'model' with your loaded model)
import torch.nn as nn
model = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # quick placeholder
try:
    with torch.no_grad():
        out = model(im_tensor)
    print("CNN forward pass OK, output shape:", out.shape)
except Exception as e:
    print("CNN forward error:", e)
