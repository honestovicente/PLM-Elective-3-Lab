import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read an image
img = cv2.imread('E:\\PLM CET SUBJECTS\\Digital Image Processing\\flower.jpg')

# Convert the image from BGR (OpenCV default) to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.show()

# Get image dimensions (rows, columns, color channels)
rows, cols, channels = img.shape
print(f'Image size: {rows} x {cols} x {channels}')

# Check color model (grayscale or RGB)
if channels == 1:
    print('Color Model: Grayscale')
else:
    print('Color Model: RGB')

# Access individual pixels (example: center pixel)
center_row = rows // 2
center_col = cols // 2
center_pixel = img[center_row, center_col, :]
print(f'Center pixel value: {center_pixel}')

# Basic arithmetic operations (add constant value to all pixels)
brightened_img = cv2.add(img, 50)
brightened_img_rgb = cv2.cvtColor(brightened_img, cv2.COLOR_BGR2RGB)
plt.figure(2)
plt.imshow(brightened_img_rgb)
plt.title('Image Brightened')
plt.show()

# Basic geometric operation (flipping image horizontally)
flipped_img = cv2.flip(img, 1)
flipped_img_rgb = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB)
plt.figure(3)
plt.imshow(flipped_img_rgb)
plt.title('Image Flipped Horizontally')
plt.show()
