img_color = imread('C:\Users\user\Downloads\orange.jpg');

% Red
img_red = img_color;
img_red(:,:,2) = 0;
img_red(:,:,3) = 0;
figure(1);
imshow(img_red);

% Green
img_green = img_color;
img_green(:,:,1) = 0;
img_green(:,:,3) = 0;
figure(2);
imshow(img_green);

% Blue
img_blue = img_color;
img_blue(:,:,1) = 0;
img_blue(:,:,2) = 0;
figure(3);
imshow(img_blue);

% Gray
g = rgb2gray(img_color);
figure(4);
imshow(g);

% Save files
imwrite(img_red, 'C:\Users\user\Downloads\img_red.jpg')
imwrite(img_green, 'C:\Users\user\Downloads\img_green.jpg')
imwrite(img_blue, 'C:\Users\user\Downloads\img_blue.jpg')
imwrite(g, 'C:\Users\user\Downloads\g.jpg')

