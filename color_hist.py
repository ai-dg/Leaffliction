# Create this simple script: test_features.py
import cv2
import numpy as np
from pathlib import Path
import sys

# Get image path from command line
if len(sys.argv) < 2:
    print("Usage: python color_hist.py <image_path>")
    sys.exit(1)

img_path = Path(sys.argv[1])
img = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Extract color histogram (simple!)
hist_r = cv2.calcHist([img_rgb], [0], None, [32], [0, 256])
hist_g = cv2.calcHist([img_rgb], [1], None, [32], [0, 256])
hist_b = cv2.calcHist([img_rgb], [2], None, [32], [0, 256])

print("Red channel histogram shape:", hist_r.shape)  # (32, 1)
print("First 5 bins:", hist_r[:5].flatten())
print("Green channel histogram shape:", hist_g.shape)  # (32, 1)
print("First 5 bins:", hist_g[:5].flatten())
print("Blue channel histogram shape:", hist_b.shape)  # (32, 1)
print("First 5 bins:", hist_b[:5].flatten())

# 2. Flatten into feature vector
features = np.concatenate([hist_r, hist_g, hist_b]).flatten()
print("\nTotal features from histogram:", features.shape)  # (96,)

# This is what we'll feed to SVM!