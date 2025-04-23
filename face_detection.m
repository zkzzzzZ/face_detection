% 定义名字列表
nameList = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10' ...
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20' ...
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'};
mainImageFilePath = "face_detection_zkz\picture_input";
mainImageSavePath = "face_detection_zkz\face_output";
% 遍历每个名字

faceDetector = buildDetector();
detector.MinSize = [30, 30]; % 检测的最小人脸尺寸
detector.MergeThreshold = 5; % 合并重叠框的阈值

for n = 1:length(nameList)
    personName = nameList{n};
    stImageFilePath = fullfile(mainImageFilePath, personName, filesep);
    dirImagePathList = dir(fullfile(stImageFilePath, '*jpg'));
    iImageNum = length(dirImagePathList);

    if iImageNum > 0

        for i = 1:iImageNum
            stImagePath = dirImagePathList(i).name;
            mImageCurrent = imread(strcat(stImageFilePath, stImagePath));

            if (size(mImageCurrent, 3) == 1) %将灰度图变成三通道图
                mImage2detect(:, :, 1) = mImageCurrent;
                mImage2detect(:, :, 2) = mImageCurrent;
                mImage2detect(:, :, 3) = mImageCurrent;
            else
                mImage2detect = mImageCurrent;
            end

            [bbox, bbimg, faces, bbfaces] = detectFaceParts(faceDetector, mImage2detect, 2);
            %imshow(bbimg)

            % 计算仿射变换矩阵
            fixed_points = [
                30.2946, 51.6963;
                65.5318, 51.5014;
                48.0252, 71.7366
            ];
            % landmarks
            moving_points = squeeze(landmarks(1,1:3,1:2));
            tform = fitgeotrans(moving_points, fixed_points, 'affine');
            
            % 应用变换
            aligned_face = imwarp(gray_face, tform, 'OutputView', imref2d(size(gray_face)));

            for l = 1:length(faces)
                imshow(faces{1})
            end

            %imcrop()
            %imwrite(bbimg, fullfile(mainImageSavePath, personName, stImagePath))
        end

    end

end
