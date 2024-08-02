import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Read an image
img = cv2.imread('E:/PLM CET SUBJECTS/Digital Image Processing/flower.jpg')

# Display the original image
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Convert to grayscale if the image is RGB
if len(img.shape) == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img

# Display the grayscale image
plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Contrast enhancement using imadjust equivalent in Python
img_contrast_enhanced = exposure.rescale_intensity(img_gray, in_range=(img_gray.min(), img_gray.max()))

# Display the contrast-enhanced image
plt.figure()
plt.imshow(img_contrast_enhanced, cmap='gray')
plt.title('Contrast Enhanced Image (imadjust)')
plt.show()

# Histogram equalization
img_histeq = cv2.equalizeHist(img_gray)

# Display the histogram equalized image
plt.figure()
plt.imshow(img_histeq, cmap='gray')
plt.title('Equalized Image')
plt.show()

# Filtering using average filter
h_avg = np.ones((5, 5), np.float32) / 25
img_avg_filtered = cv2.filter2D(img_gray, -1, h_avg)

# Display the average filtered image
plt.figure()
plt.imshow(img_avg_filtered, cmap='gray')
plt.title('Filtered Image (Average)')
plt.show()

# Filtering using median filter
img_median_filtered = cv2.medianBlur(img_gray, 5)

# Display the median filtered image
plt.figure()
plt.imshow(img_median_filtered, cmap='gray')
plt.title('Filtered Image (Median)')
plt.show()

# Display histograms for comparison
fig, axes = plt.subplots(3, 2, figsize=(12, 8))

axes[0, 0].hist(img_gray.ravel(), bins=256, histtype='step', color='black')
axes[0, 0].set_title('Histogram of Grayscale')

axes[0, 1].imshow(img_gray, cmap='gray')
axes[0, 1].set_title('Grayscale Image')

axes[1, 0].hist(img_contrast_enhanced.ravel(), bins=256, histtype='step', color='black')
axes[1, 0].set_title('Histogram of Enhanced Image')

axes[1, 1].imshow(img_contrast_enhanced, cmap='gray')
axes[1, 1].set_title('Contrast Enhanced Image')

axes[2, 0].hist(img_histeq.ravel(), bins=256, histtype='step', color='black')
axes[2, 0].set_title('Histogram of Equalized Image')

axes[2, 1].imshow(img_histeq, cmap='gray')
axes[2, 1].set_title('Equalized Image')

fig.tight_layout()
plt.show()

# Display histograms for filtered images
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(img_avg_filtered.ravel(), bins=256, histtype='step', color='black')
axes[0, 0].set_title('Histogram of Average Filtered')

axes[0, 1].imshow(img_avg_filtered, cmap='gray')
axes[0, 1].set_title('Average Filtered Image')

axes[1, 0].hist(img_median_filtered.ravel(), bins=256, histtype='step', color='black')
axes[1, 0].set_title('Histogram of Median Filtered')

axes[1, 1].imshow(img_median_filtered, cmap='gray')
axes[1, 1].set_title('Median Filtered Image')

fig.tight_layout()
plt.show()
