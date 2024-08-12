import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage import color, filters

# Load the image
img = cv2.cvtColor(cv2.imread('flower.jpg'), cv2.COLOR_RGB2BGR) # matplotlib will read the image in RGB format

# Convert to grayscale if the image is BGR
if img.shape[2] == 3:
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    grayImg = img

# Thresholding Using Otsu's Method
_, bw = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # MATLAB: graythresh() and imbinarize(, level)

plt.figure(1)
plt.subplot(1, 2, 1), plt.imshow(img), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(bw, cmap='gray'), plt.title('Binary Image')
plt.show()

# Multi-Level Thresholding Using Otsu's Method
numCluster = 3
kMeans = KMeans(n_clusters=numCluster, random_state=0).fit(grayImg.reshape(-1, 1))
segImg = kMeans.labels_.reshape(grayImg.shape)
segImg = np.uint8(segImg * 255 / (numCluster - 1))  # MATLAB: multithresh() and imquantize(, level)

plt.figure(2)
plt.subplot(1, 2, 1), plt.imshow(img), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(segImg, cmap='gray'), plt.title('Segmented Image')
plt.show()

# Global Histogram Thresholding Using Otsu's Method
counts, bins = np.histogram(grayImg.flatten(), bins=16, range=(0, 255))  # MATLAB: imhist(, 16)
otsuThresh = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]  # MATLAB: otsuthresh(counts)
_, bwOtsu = cv2.threshold(grayImg, otsuThresh, 255, cv2.THRESH_BINARY)  # MATLAB: imbinarize(, T)

plt.figure(3)
plt.imshow(bwOtsu, cmap='gray')
plt.title('Binary Image')
plt.show()

# Region-Based Segmentation Using K-Means
bwImg = cv2.cvtColor(cv2.imread('flower.jpg'), cv2.COLOR_BGR2GRAY)

kMeans = KMeans(n_clusters=3, random_state=0).fit(bwImg.reshape(-1, 1))  # MATLAB: imsegkmeans(, 3)
labels = kMeans.labels_.reshape(bwImg.shape)
labelOverlay = cv2.applyColorMap(np.uint8(labels * 255 / 2), cv2.COLORMAP_JET)  # MATLAB: labeloverlay(, L)

plt.figure(4)
plt.imshow(labelOverlay)
plt.title('Labeled Image')
plt.show()

# Connected-Component Labeling
_, binImg2 = cv2.threshold(bwImg, otsuThresh, 255, cv2.THRESH_BINARY)  # MATLAB: imbinarize()
numLabels, labeledImg = cv2.connectedComponents(binImg2)  # MATLAB: bwlabel(bin_img2)
coloredLabels = cv2.applyColorMap(np.uint8(labeledImg * 255 / numLabels), cv2.COLORMAP_JET)  # MATLAB: label2rgb(labeledImage, 'hsv', 'k', 'shuffle')

print('Number of connected components: ', numLabels)  # MATLAB: disp(['Number of connected components: ', num2str(numberOfComponents)])

plt.figure(5)
plt.imshow(coloredLabels)
plt.title('Labeled Image')
plt.show()

# Adding Noise and Segmentation
noiseImg = np.clip(grayImg + np.random.normal(0, 25, grayImg.shape), 0, 255).astype(np.uint8)  # MATLAB: imnoise(, 'salt & pepper', 0.09)
otsuThreshNoise = filters.threshold_otsu(noiseImg)  # MATLAB: multithresh(img_noise)
_, segImgNoise = cv2.threshold(noiseImg, otsuThreshNoise, 255, cv2.THRESH_BINARY)  # MATLAB: imbinarize(, level)

plt.figure(6)
plt.subplot(1, 2, 1), plt.imshow(noiseImg, cmap='gray'), plt.title('Noisy Image')
plt.subplot(1, 2, 2), plt.imshow(segImgNoise, cmap='gray'), plt.title('Segmented Image with noise')
plt.show()

# Segmenting the image into two regions using K-Means clustering
kMeans = KMeans(n_clusters=2, random_state=0).fit(img.reshape(-1, 3))  # MATLAB: imsegkmeans(RGB, 2)
labels = kMeans.labels_.reshape(img.shape[:2])
labelOverlay = cv2.applyColorMap(np.uint8(labels * 255 / 2), cv2.COLORMAP_JET)  # MATLAB: labeloverlay(RGB, L)

plt.figure(7)
plt.imshow(labelOverlay)
plt.title('Labeled Image')
plt.show()

# Creating and Applying Gabor Filters
def gaborFilter(img, wavelength, orientation):
    filters = []
    for theta in orientation:
        theta = np.deg2rad(theta)
        for lambda_ in wavelength:
            kernel = cv2.getGaborKernel((31, 31), 4.0, theta, lambda_, 0.5, 0, cv2.CV_32F)  # MATLAB: gabor(wavelength, orientation)
            filters.append(kernel)
    return filters

wavelength = [2 ** i * 3 for i in range(6)]  # MATLAB: 2.^(0:5) * 3
orientation = list(range(0, 180, 45))  # MATLAB: 0:45:135
gaborKernels = gaborFilter(grayImg, wavelength, orientation)

gaborMag = np.zeros_like(grayImg, dtype=np.float32)
for kernel in gaborKernels:
    filteredImg = cv2.filter2D(grayImg, cv2.CV_32F, kernel)
    gaborMag = np.maximum(gaborMag, np.abs(filteredImg))

plt.figure(8)
num_kernels = len(gaborKernels)
for i in range(num_kernels):
    plt.subplot(4, 6, i + 1)
    plt.imshow(cv2.filter2D(grayImg, cv2.CV_32F, gaborKernels[i]), cmap='gray')
plt.suptitle('Gabor Filtered Images')
plt.show()

# Smoothing Gabor Filtered Images
for i, kernel in enumerate(gaborKernels):
    sigma = 0.5 * wavelength[i % len(wavelength)]
    gaborMag = cv2.GaussianBlur(gaborMag, (0, 0), sigma)  # MATLAB: imgaussfilt(gabormag(:, :, i), 3 * sigma)

plt.figure(9)
for i in range(num_kernels):
    plt.subplot(4, 6, i + 1)
    plt.imshow(gaborMag, cmap='gray')
plt.suptitle('Smoothed Gabor Filtered Images')
plt.show()

# Feature Set for Clustering
x, y = np.meshgrid(np.arange(grayImg.shape[1]), np.arange(grayImg.shape[0]))  # MATLAB: meshgrid(1:ncols, 1:nrows)
featureSet = np.stack([grayImg, gaborMag, x, y], axis=-1)  # MATLAB: cat(3, bw_RGB, gabormag, X, Y)

featureSetReshaped = featureSet.reshape(-1, featureSet.shape[-1])
kMeans = KMeans(n_clusters=2, random_state=0).fit(featureSetReshaped)  # MATLAB: imsegkmeans(featureSet, 2, 'NormalizeInput', true)
labels = kMeans.labels_.reshape(grayImg.shape)
labelOverlay = color.label2rgb(labels, image=img, bg_label=0)  # MATLAB: label2rgb(RGB, L2)

plt.figure(10)
plt.imshow(labelOverlay)
plt.title('Labeled Image with Additional Pixel Information')
plt.show()