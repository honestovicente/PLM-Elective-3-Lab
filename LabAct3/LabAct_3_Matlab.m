
% Read an image
img = imread('C:\Users\user\Downloads\LabAct3\flower.jpg');

% Display the original image
figure;
imshow(img);
title('Original Image');

% Convert to grayscale if the image is RGB
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

% Display the grayscale image
figure;
imshow(img_gray);
title('Grayscale Image');

% Contrast enhancement using imadjust
img_contrast_enhanced = imadjust(img_gray);

% Display the contrast-enhanced image
figure;
imshow(img_contrast_enhanced);
title('Contrast Enhanced Image (imadjust)');

% Histogram equalization
img_histeq = histeq(img_gray);

% Display the histogram equalized image
figure;
imshow(img_histeq);
title('Equalized Image');

% Filtering using average filter
h_avg = fspecial('average', [5, 5]);
img_avg_filtered = imfilter(img_gray, h_avg);

% Display the average filtered image
figure;
imshow(img_avg_filtered);
title('Filtered Image (Average)');

% Filtering using median filter
img_median_filtered = medfilt2(img_gray, [5, 5]);

% Display the median filtered image
figure;
imshow(img_median_filtered);
title('Filtered Image (Median)');

% Display histograms for comparison

% Grayscale histogram
figure;
imhist(img_gray);
title('Histogram of Grayscale Image');

% Enhanced histogram (imadjust)
figure;
imhist(img_contrast_enhanced);
title('Histogram of Enhanced Image (imadjust)');

% Equalized histogram
figure;
imhist(img_histeq);
title('Histogram of Equalized Image');

% Histogram (Average Filtered)
figure;
imhist(img_avg_filtered);
title('Histogram of Average Filtered Image');

% Histogram (Median Filtered)
figure;
imhist(img_median_filtered);
title('Histogram of Median Filtered Image');

% --------MODIFICATIONS-----------

% Convert to grayscale if the image is RGB
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

% Filtering using average filter but different values
h_avg = fspecial('average', [10, 10]); % Original is [5, 5]
img_avg_filtered_2 = imfilter(img_gray, h_avg);

% Show the experimented image
figure;
imshow(img_avg_filtered_2);
title('Filtered Image (Using Average but Different values)');

% Filtering using median filter
img_median_filtered_2 = medfilt2(img_gray, [1, 10]); % Original is [5, 5]

% Display the median filtered image
figure;
imshow(img_median_filtered_2);
title('Experimented Filtered Image (Median)');

% Show the Histogram
figure;
imhist(img_median_filtered_2);
title('Histogram of Experimented Median Filtered Image');

% Save images

imwrite(img_gray, 'C:\Users\user\Downloads\LabAct3\matlab\img_gray.jpg');
imwrite(img_contrast_enhanced, 'C:\Users\user\Downloads\LabAct3\matlab\img_contrast_enhanced.jpg');
imwrite(img_histeq, 'C:\Users\user\Downloads\LabAct3\matlab\img_histeq.jpg');
imwrite(img_avg_filtered, 'C:\Users\user\Downloads\LabAct3\matlab\img_avg_filtered.jpg');
imwrite(img_median_filtered, 'C:\Users\user\Downloads\LabAct3\matlab\img_median_filtered.jpg');

imwrite(img_avg_filtered_2, 'C:\Users\user\Downloads\LabAct3\matlab\img_avg_filtered_2.jpg');
imwrite(img_median_filtered_2, 'C:\Users\user\Downloads\LabAct3\matlab\img_median_filtered_2.jpg');
