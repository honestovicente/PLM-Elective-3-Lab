import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
img = cv2.imread('E:/PLM CET SUBJECTS/Digital Image Processing/flower.jpg')

# Convert to grayscale if the image is RGB
if len(img.shape) == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img

# Filtering using average filter but different values
h_avg = np.ones((10, 10), np.float32) / 100  # Original is [5,5]
img_avg_filtered = cv2.filter2D(img_gray, -1, h_avg)

# Show the experimented average filtered image
plt.figure()
plt.imshow(img_avg_filtered, cmap='gray')
plt.title('Filtered Image (Using Average but Different values)')
plt.show()

# Filtering using median filter with separate dimensions
img_median_filtered = cv2.medianBlur(img_gray, ksize=1)  # Apply a median filter with ksize=1 (no change)
img_median_filtered = cv2.medianBlur(img_median_filtered, ksize=11)  # Apply a median filter with ksize=11 for the effect

# Display the median filtered image
plt.figure()
plt.imshow(img_median_filtered, cmap='gray')
plt.title('Experimented Filtered Image (Median)')
plt.show()

# Show the Histogram of the experimented median filtered image
plt.figure()
plt.hist(img_median_filtered.ravel(), bins=256, histtype='step', color='black')
plt.title('Histogram of Experimented Median Filtered')
plt.show()
