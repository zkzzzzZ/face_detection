%% Initial
clear;
clc;
img_num=2;  %the number of image
compress_scale=0.4;  %define image compress scale
points_p=[0 0;1 0;2 1;2 0]; %define cordinate of 2D plain in 3D space


%% define a cell that load image

Image=cell(1,img_num);

%% read the image
Image{1,1}=imread('book1.jpg');
Image{1,2}=imread('book2.jpg');

%% image compression, transform rgb to gray, and select feature points
feature=[];
for i=1:img_num
    Image{1,i}=imresize(Image{1,i},compress_scale);
    I{:,:,i}=Image{1,i};
    Image{1,i}=rgb2gray(Image{1,i});
    imshow(Image{1,i});
    hold on;
    for j=1:4
        [x,y]= ginput(1);        %select the corner
        x=round(x);
        y=round(y);
        plot(x,y,'ro');
        feature(j,2*i-1)=x;     %feature is a matrix containing corner cordination
        feature(j,2*i)=y;
    end
    close all;
end

%% calculate homegraphy matrix for each matrix
featurep1=feature(:,1:2);
featurep2=feature(:,3:4);
h = calc_homography(featurep2, featurep1);
Im=I{:,:,2};
h

[a,b]=size(I);
tform=projective2d(h);
J=imwarp(Im,tform);    % matlab自带的处理图像变换的函数
figure,imshow(I{:,:,1});
figure,imshow(I{:,:,2});
figure,imshow(J)


%% calculate homegraphy matrix for each matrix
function T = calc_homography(points1, points2)

xaxb = points2(:,1) .* points1(:,1);
xayb = points2(:,1) .* points1(:,2);
yaxb = points2(:,2) .* points1(:,1);
yayb = points2(:,2) .* points1(:,2);

A = zeros(size(points1, 1)*2, 9);
A(1:2:end,3) = 1;
A(2:2:end,6) = 1;
A(1:2:end,1:2) = points1;
A(2:2:end,4:5) = points1;
A(1:2:end,7) = -xaxb;
A(1:2:end,8) = -xayb;
A(2:2:end,7) = -yaxb;
A(2:2:end,8) = -yayb;
A(1:2:end,9) = -points2(:,1);
A(2:2:end,9) = -points2(:,2);

[~,~,V] = svd(A);
h = V(:,9) ./ V(9,9);
T= reshape(h,3,3);
end