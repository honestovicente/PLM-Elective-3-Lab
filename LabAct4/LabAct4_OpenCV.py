import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
from scipy.ndimage import gaussian_filter, median_filter


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


# Gaussian filtering
img_gaussian_filtered = gaussian_filter(img_blur, sigma=1)


# Display the Gaussian filtered image
plt.figure()
plt.imshow(img_gaussian_filtered, cmap='gray')
plt.title('Filtered Image (Gaussian)')
plt.axis('off')
plt.show()


# Sharpening using unsharp masking
img_sharpened = cv2.addWeighted(img_blur, 1.5, cv2.GaussianBlur(img_blur, (0, 0), 1), -0.5, 0)


# Display the sharpened image
plt.figure()
plt.imshow(img_sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')
plt.show()


# Add Gaussian noise and remove it using median filter
img_noisy = img_gray + np.random.normal(0, 25, img_gray.shape)
img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
img_noisy_removed = median_filter(img_noisy, size=5)


# Display the noisy image
plt.figure()
plt.imshow(img_noisy, cmap='gray')
plt.title('Noisy')
plt.axis('off')
plt.show()


# Display the noise-removed image
plt.figure()
plt.imshow(img_noisy_removed, cmap='gray')
plt.title('Noise Removed')
plt.axis('off')
plt.show()


# Deblurring
estimated_nsr = 0.01
img_deblurred = restoration.wiener(img_blur, psf, estimated_nsr)


# Display the deblurred image
plt.figure()
plt.imshow(img_deblurred, cmap='gray')
plt.title('Deblurred Image')
plt.axis('off')
plt.show()
