import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from skimage import restoration


# Read the image
img = cv2.imread('flower.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Display the original image
plt.figure()
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()


# Convert to grayscale if the image is RGB
if len(img.shape) == 3:
   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
   img_gray = img


# Display the grayscale image
plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')
plt.show()


# Add blur to the image
len = 21
theta = 11
psf = np.zeros((len, len))
psf[len//2, :] = 1
psf = cv2.warpAffine(psf, cv2.getRotationMatrix2D((len/2, len/2), theta, 1.0), (len, len))
psf = psf / psf.sum()
img_blur = cv2.filter2D(img_gray, -1, psf)


# Show the blurred image
plt.figure()
plt.imshow(img_blur, cmap='gray')
plt.title('Motion Blurred Image')
plt.axis('off')
plt.show()


# Gaussian filtering with different parameters
h_gaussian = gaussian_filter(img_gray, sigma=10)
img_gaussian_filtered = cv2.filter2D(img_gray, -1, h_gaussian)


# Display the Gaussian filtered image
plt.figure()
plt.imshow(img_gaussian_filtered, cmap='gray')
plt.title('Filtered Image with Experimented Value (Gaussian)')
plt.axis('off')
plt.show()


# Display the histogram of the Gaussian filtered image
plt.figure()
plt.hist(img_gaussian_filtered.ravel(), bins=256, fc='k', ec='k')
plt.title('Histogram of the Experimented Value (Gaussian Filtered)')
plt.show()


# Add Gaussian noise with different values
img_noisy_exp1 = img_gray + np.random.normal(0, 0.5 * 255, img_gray.shape).astype(np.uint8)
img_noisy_exp2 = img_gray + np.random.normal(0, 0.1 * 255, img_gray.shape).astype(np.uint8)
img_noisy_exp1 = np.clip(img_noisy_exp1, 0, 255).astype(np.uint8)
img_noisy_exp2 = np.clip(img_noisy_exp2, 0, 255).astype(np.uint8)


# Display the noisy images
plt.figure()
plt.imshow(img_noisy_exp1, cmap='gray')
plt.title('Noisy Using Experimented Value (Gaussian is 0.5)')
plt.axis('off')
plt.show()


plt.figure()
plt.imshow(img_noisy_exp2, cmap='gray')
plt.title('Noisy Using Experimented Value (Gaussian is 0.1)')
plt.axis('off')
plt.show()


# Display the histograms for the noisy images
plt.figure()
plt.hist(img_noisy_exp1.ravel(), bins=256, fc='k', ec='k')
plt.title('Histogram of Noisy Image Experimented Value 1')
plt.show()


plt.figure()
plt.hist(img_noisy_exp2.ravel(), bins=256, fc='k', ec='k')
plt.title('Histogram of Noisy Image Experimented Value 2')
plt.show()
