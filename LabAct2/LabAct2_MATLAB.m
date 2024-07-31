% Read an image
img = imread('C:\Users\crist\Downloads\flower.jpg');

% Display the original image
figure(1);
imshow(img);
title('Original Image');

% Get image dimensions (rows, columns, color channels)
[rows, cols, channels] = size(img);
disp(['Image size: ', num2str(rows), ' x ', num2str(cols), ' x ', num2str(channels)]);

% Check color model (grayscale or RGB)
if channels == 1
    disp('Color Model: Grayscale');
else
    disp('Color Model: RGB');
end

% Access individual pixels (example: center pixel)
center_row = floor(rows / 2) + 1;
center_col = floor(cols / 2) + 1;
center_pixel = img(center_row, center_col, :);
disp(['Center pixel value: ', num2str(center_pixel(:)')]);

% Basic arithmetic operations (add constant value to all pixels)
brightened_img = img + 50;
figure(2);
imshow(brightened_img);
title('Image Brightened');

% Basic geometric operation (flipping image horizontally)
flipped_img = fliplr(img);
figure(3);
imshow(flipped_img);
title('Image Flipped Horizontally');