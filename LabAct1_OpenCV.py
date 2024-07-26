import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('E:/PLM CET SUBJECTS/Digital Image Processing/flower.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Rotate by 30 degrees
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated_img = cv2.warpAffine(img, M, (w, h))

# Flip horizontally
flipped_img = cv2.flip(rotated_img, 1)

# Display results
plt.figure(1)
plt.imshow(img)
plt.title('Original Image')

plt.figure(2)
plt.imshow(rotated_img)
plt.title('Rotated 30°')

plt.figure(3)
plt.imshow(flipped_img)
plt.title('Rotated & Flipped')

plt.show()