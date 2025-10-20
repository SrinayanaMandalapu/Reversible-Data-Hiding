# test_med_1002.py
import os
import imageio.v2 as imageio
from med import med_predictor  # replace with your actual import

path = os.path.join("BOSSbase_1.01","1002.pgm")
im = imageio.imread(path)

try:
    result = med_predictor(im)  # whatever your function signature is
    print("Med predictor OK, result type/shape:", type(result), getattr(result, 'shape', None))
except Exception as e:
    print("Med predictor error:", e)
