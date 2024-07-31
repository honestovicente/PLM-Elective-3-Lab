pkg load image

% Read an image
img = imread('C:\Users\user\Downloads\IMG Color\flower.jpg');

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

% Other functionalities

% Individual color channels

% Red
img_red = img;
img_red(:,:,2) = 0;
img_red(:,:,3) = 0;
figure(4);
imshow(img_red);
title('Red');

% Green
img_green = img;
img_green(:,:,1) = 0;
img_green(:,:,3) = 0;
figure(5);
imshow(img_green);
title('Green');

% Blue
img_blue = img;
img_blue(:,:,1) = 0;
img_blue(:,:,2) = 0;
figure(6);
imshow(img_blue);
title('Blue');

% Other arithmetic operations

% Darken Image by subtraction
darkened_img = img - 50;
figure(8)
imshow(darkened_img);
title('Darkened Image');

% Brightness increase by multiplying to 4
more_brightened_img = img * 4;
figure(9)
imshow(more_brightened_img);
title('Brightness multiplied increase');

% Other geometric operation (rotate)
rotated_img = imrotate(img,50);
figure(10);
imshow(rotated_img);
title('Rotate image by 50 degrees');

imwrite(brightened_img, 'C:\Users\user\Downloads\IMG Color\brightened_img.jpg');
imwrite(flipped_img, 'C:\Users\user\Downloads\IMG Color\flipped_img.jpg');
imwrite(img_red, 'C:\Users\user\Downloads\IMG Color\img_red.jpg');
imwrite(img_green, 'C:\Users\user\Downloads\IMG Color\img_green.jpg');
imwrite(img_blue, 'C:\Users\user\Downloads\IMG Color\img_blue.jpg');
imwrite(darkened_img, 'C:\Users\user\Downloads\IMG Color\darkened_img.jpg');
imwrite(more_brightened_img, 'C:\Users\user\Downloads\IMG Color\more_brightened_img.jpg');
imwrite(rotated_img, 'C:\Users\user\Downloads\IMG Color\rotated_img.jpg');
