function detect2()
% 初始化人脸检测器
detector = buildDetector();
detector.MinSize = [50, 50]; % 检测的最小人脸尺寸
detector.MergeThreshold = 3; % 合并重叠框的阈值

% 设置路径
inputDir = 'faces';
outputDir = 'output';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 获取所有图片文件（支持jpg/jpeg/png不区分大小写）
fileExtensions = {'*.jpg', '*.jpeg', '*.png'};
fileList = [];

for i = 1:length(fileExtensions)
    fileList = [fileList; dir(fullfile(inputDir, fileExtensions{i}))]; %#ok<AGROW>
end

% 处理每张图片
for i = 1:length(fileList)
    fileName = fileList(i).name;
    filePath = fullfile(inputDir, fileName);

    try
        fileName
        % 读取图像并灰度化
        img = imread(filePath);
        %img = imresize(img, [1680, 1680], 'Method', 'bilinear');
        
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        % 检测人脸
        [bboxes, ~, faces, ~] = detectFaceParts(detector, img); % 修改为不使用bbfaces

        if isempty(faces)
            continue;
        end

        % 找出尺寸最大的人脸
        maxFaceArea = 0;
        maxFaceIndex = 0;

        for j = 1:length(faces)
            currentBox = bboxes(j, :);
            currentArea = currentBox(3) * currentBox(4); % 计算当前人脸的面积（宽*高）

            if currentArea > maxFaceArea
                maxFaceArea = currentArea;
                maxFaceIndex = j;
            end
        end

        % 只处理最大的人脸
        if maxFaceIndex > 0
            face = faces{maxFaceIndex};
            % bbface = bbfaces{maxFaceIndex};
            bbox = bboxes(maxFaceIndex, :); % 获取最大人脸的边界框

            scaleFactorX = 512 / bbox(3); % 计算宽度的缩放比例
            scaleFactorY = 512 / bbox(4); % 计算高度的缩放比例

            % bbox

            for i = 1:4
                bbox(1 + 4 * i) = round((bbox(1 + 4 * i) - bbox(1)) * scaleFactorX); % 缩放边界框的x坐标
                bbox(2 + 4 * i) = round((bbox(2 + 4 * i) - bbox(2)) * scaleFactorY); % 缩放边界框的y坐标
                bbox(3 + 4 * i) = round(bbox(3 + 4 * i) * scaleFactorX); % 缩放边界框的宽度
                bbox(4 + 4 * i) = round(bbox(4 + 4 * i) * scaleFactorY); % 缩放边界框的高度
            end

            resizedFace = imresize(face, [512, 512], 'Method', 'bilinear');
            alignedFace = alignFace(resizedFace, bbox); % 对齐人脸
            alignedFace = imresize(alignedFace, [353, 353], 'Method', 'bilinear');

            % 生成输出文件名
            [~, name] = fileparts(fileName);
            outputName = sprintf('%s.jpg', name);
            outputPath = fullfile(outputDir, outputName);

            % 保存图像
            imwrite(alignedFace, outputPath);
        end

    catch ME
        fprintf('处理文件 %s 出错: %s\n', fileName, ME.message);
    end

end

disp('处理完成！');
end
