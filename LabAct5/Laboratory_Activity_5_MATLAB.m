% Global Image Thresholding Using Otsu's Method

% Load the image
img = imread('flower.jpg'); % Changed from 'original image' to 'flower.jpg' to match the context

% Convert to grayscale if the image is not already grayscale
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

% Calculate threshold using graythresh
level = graythresh(img_gray);

% Convert into binary image using the computed threshold
bw = imbinarize(img_gray, level);

% Display the original image and the binary image
figure(1);
imshowpair(img, bw, 'montage');
title('Original Image (left) and Binary Image (right)');

% Multi-Level Thresholding Using Otsu's Method

% Calculate multiple thresholds using multithresh
level = multithresh(img_gray);

% Segment the image into regions using imquantize
seg_img = imquantize(img_gray, level);

% Display the original image and the segmented image
figure(2);
imshowpair(img, seg_img, 'montage');
title('Original Image (left) and Segmented Image (right)');

% Global Histogram Thresholding Using Otsu's Method

% Compute a global threshold using the histogram counts
[counts, x] = imhist(img_gray, 16); % Compute histogram counts
T = otsuthresh(counts); % Compute global threshold
bw = imbinarize(img_gray, T); % Create binary image using the computed threshold

% Display the binary image
figure(3);
imshow(bw);
title('Binary Image');

% Region-Based Segmentation

% Using K-means clustering
img2 = imread('flower.jpg');

% Convert the image to grayscale
bw_img2 = rgb2gray(img2);

% Segment the image into three regions using k-means clustering
[L, centers] = imsegkmeans(bw_img2, 3);
B = labeloverlay(bw_img2, L);

figure(4);
imshow(B);
title('Labeled Image');

% Using Connected-Component Labeling
bin_img2 = imbinarize(bw_img2);

% Label the connected components
[labeledImage, numberOfComponents] = bwlabel(bin_img2);

% Display the number of connected components
disp(['Number of connected components: ', num2str(numberOfComponents)]);

% Assign a different color to each connected component
coloredLabels = label2rgb(labeledImage, 'hsv', 'k', 'shuffle');

% Display the labeled image
figure(5);
imshow(coloredLabels);
title('Labeled Image');

% Parameter Modifications

% Adding noise to the image then segmenting it using Otsu's method
img_noise = imnoise(img_gray, 'salt & pepper', 0.09);

% Calculate thresholds using multithresh
level = multithresh(img_noise);

% Segment the image into regions using imquantize
seg_img = imquantize(img_noise, level);

% Display the original image and the segmented image with noise
figure(6);
imshowpair(img_noise, seg_img, 'montage');
title('Original Image (left) and Segmented Image with noise (right)');

% Segment the image into two regions using k-means clustering
RGB = imread('flower.jpg');
L = imsegkmeans(RGB, 2);
B = labeloverlay(RGB, L);

figure(7);
imshow(B);
title('Labeled Image');

% Create a set of 24 Gabor filters
wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
g = gabor(wavelength, orientation);

% Convert the image to grayscale
bw_RGB = im2gray(im2single(RGB));

% Filter the grayscale image using the Gabor filters
gabormag = imgaborfilt(bw_RGB, g);

% Display the 24 filtered images in a montage
figure(8);
montage(gabormag, 'Size', [4 6]);

% Smooth each filtered image to remove local variations
for i = 1:length(g)
    sigma = 0.5 * g(i).Wavelength;
    gabormag(:, :, i) = imgaussfilt(gabormag(:, :, i), 3 * sigma);
end

% Display the smoothed images in a montage
figure(9);
montage(gabormag, 'Size', [4 6]);

% Get the x and y coordinates of all pixels in the input image
[nrows, ncols, ~] = size(RGB);
[X, Y] = meshgrid(1:ncols, 1:nrows);
featureSet = cat(3, bw_RGB, gabormag, X, Y);

% Segment the image into two regions using k-means clustering with the supplemented feature set
L2 = imsegkmeans(featureSet, 2, 'NormalizeInput', true);
C = labeloverlay(RGB, L2);

figure(10);
imshow(C);
title('Labeled Image with Additional Pixel Information');
