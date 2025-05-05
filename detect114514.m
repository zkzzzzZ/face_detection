% 初始化人脸检测器
detector = buildDetector();
detector.MinSize = [50, 50]; % 检测的最小人脸尺寸
detector.MergeThreshold = 3; % 合并重叠框的阈值

% 人脸对齐函数
function alignedFace = alignFace(face, bbox)
    % 提取关键点坐标
    landmarks = []; % 初始化为空数组
    targetPoints = []; % 初始化目标点数组
    target = [
        73   152
        253   274
        283   155
        452   269
        172   402
        344   504
        169   233
        361   396
    ];

    for i = 1:4
        x1 = bbox(1 + 4 * i);
        y1 = bbox(2 + 4 * i);
        x2 = x1 + bbox(3 + 4 * i);
        y2 = y1 + bbox(4 + 4 * i);
        if x1 >= 0 && y1 >= 0
            landmarks = [landmarks; x1, y1; x2, y2]; % 添加有效的关键点
            targetPoints = [targetPoints; target(2 * i - 1, :); target(2 * i, :)]; % 添加对应的目标点
        end
    end

    % 计算仿射变换矩阵
    transformationMatrix = fitgeotrans(landmarks, targetPoints, 'projective');
    outputView = imref2d(size(face));
    alignedFace = imwarp(face, transformationMatrix, 'OutputView', outputView);
end

% PCA 人脸识别函数
function detect3(alignedFace, trainDir, detector)
    % 参数设置
    imgSize = [353, 353]; % 图像尺寸
    totalImages = 10; % 训练数据总共10张图像

    % 读取训练数据
    trainData = [];
    trainLabels = [];
    fileList = dir(fullfile(trainDir, '*.bmp')); % 动态获取所有bmp文件
    if length(fileList) < totalImages
        fprintf('警告：训练数据数量不足，期望 %d 张，实际找到 %d 张\n', totalImages, length(fileList));
    end

    imgCount = 0; % 已加载的包含人脸的图像计数
    for i = 1:min(totalImages, length(fileList))
        fileName = fileList(i).name;
        filePath = fullfile(trainDir, fileName);
        if exist(filePath, 'file')
            img = imread(filePath);
            if size(img, 3) == 3
                img = rgb2gray(img);
            end

            % 检测人脸
            [bboxes, ~, faces, ~] = detectFaceParts(detector, img);
            if isempty(faces)
                fprintf('训练图像 %s 未检测到人脸，跳过\n', fileName);
                continue;
            end

            % 仅使用最大的人脸
            maxFaceArea = 0;
            maxFaceIndex = 0;
            for j = 1:length(faces)
                currentBox = bboxes(j, :);
                currentArea = currentBox(3) * currentBox(4);
                if currentArea > maxFaceArea
                    maxFaceArea = currentArea;
                    maxFaceIndex = j;
                end
            end

            if maxFaceIndex == 0
                fprintf('训练图像 %s 未检测到有效人脸，跳过\n', fileName);
                continue;
            end

            face = faces{maxFaceIndex};
            bbox = bboxes(maxFaceIndex, :);
            scaleFactorX = 512 / bbox(3);
            scaleFactorY = 512 / bbox(4);
            for k = 1:4
                bbox(1 + 4 * k) = round((bbox(1 + 4 * k) - bbox(1)) * scaleFactorX);
                bbox(2 + 4 * k) = round((bbox(2 + 4 * k) - bbox(2)) * scaleFactorY);
                bbox(3 + 4 * k) = round(bbox(3 + 4 * k) * scaleFactorX);
                bbox(4 + 4 * k) = round(bbox(4 + 4 * k) * scaleFactorY);
            end

            resizedFace = imresize(face, [512, 512], 'Method', 'bilinear');
            alignedTrainFace = alignFace(resizedFace, bbox);
            alignedTrainFace = imresize(alignedTrainFace, imgSize, 'Method', 'bilinear');

            trainData = [trainData; double(alignedTrainFace(:))']; % 展平为向量
            trainLabels = [trainLabels; i];
            imgCount = imgCount + 1;
        else
            fprintf('训练文件 %s 不存在\n', filePath);
        end
    end

    % 检查训练数据是否为空
    if isempty(trainData)
        error('未加载到任何包含人脸的训练数据，请检查训练数据路径：%s', trainDir);
    end

    % 调试：显示训练数据尺寸
    fprintf('成功加载 %d 张包含人脸的训练图像\n', imgCount);
    fprintf('训练数据尺寸: %d 行, %d 列\n', size(trainData, 1), size(trainData, 2));

    % 计算均值和中心化
    mA = mean(trainData);
    centeredData = trainData - mA;

    % 计算协方差矩阵并进行特征分解
    covMatrix = centeredData * centeredData' / (size(centeredData, 1) - 1);
    [V, D] = eig(covMatrix);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx); % 按特征值降序排序

    % 选择前 k 个主成分（动态调整 k）
    k = min(10, size(V, 2)); % k 不超过训练数据数量和特征维度
    V = V(:, 1:k);

    % 计算训练数据的 PCA 投影
    pcatrainface = centeredData * V;
    fprintf('pcatrainface 尺寸: %d 行, %d 列\n', size(pcatrainface, 1), size(pcatrainface, 2));

    % 处理测试图像
    testData = double(alignedFace(:))';
    pcaB = (testData - mA) * V; % 投影到特征脸空间
    fprintf('pcaB 尺寸: %d 行, %d 列\n', size(pcaB, 1), size(pcaB, 2));

    % 寻找最匹配的训练图像
    k = norm(pcatrainface(1, :) - pcaB);
    n = 1;
    for j = 1:size(pcatrainface, 1)
        d = norm(pcatrainface(j, :) - pcaB);
        if d < k
            k = d;
            n = j;
        end
    end

    % 显示结果
    matchFile = fileList(n).name;
    matchPath = fullfile(trainDir, matchFile);
    if exist(matchPath, 'file')
        matchImg = imread(matchPath);
        if size(matchImg, 3) == 3
            matchImg = rgb2gray(matchImg);
        end

        figure;
        subplot(1, 2, 1);
        imshow(alignedFace, []);
        title('测试图像');
        subplot(1, 2, 2);
        imshow(matchImg, []);
        title(sprintf('匹配图像: %s', matchFile));
    else
        fprintf('匹配文件 %s 不存在\n', matchPath);
    end
end

% 主程序
% 设置路径
inputDir = 'C:\Users\28713\Desktop\线性代数\face_detection_zkz\picture_input\face_new\face_new\陈祺琛';
outputDir = 'C:\Users\28713\Desktop\线性代数\face_detection_zkz\registor_output';
trainDir = 'C:\Users\28713\Desktop\线性代数\face_detection_zkz\pro_output'; % 训练数据集路径

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 获取所有图片文件
fileExtensions = {'*.jpg', '*.jpeg', '*.png'};
fileList = [];
for i = 1:length(fileExtensions)
    fileList = [fileList; dir(fullfile(inputDir, fileExtensions{i}))];
end

% 处理每张图片
for i = 1:length(fileList)
    fileName = fileList(i).name;
    filePath = fullfile(inputDir, fileName);

    try
        fprintf('处理文件: %s\n', fileName);
        % 读取图像并灰度化
        img = imread(filePath);
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        % 检测人脸
        [bboxes, ~, faces, ~] = detectFaceParts(detector, img);

        if isempty(faces)
            fprintf('未检测到人脸: %s\n', fileName);
            continue;
        end

        % 找出尺寸最大的人脸
        maxFaceArea = 0;
        maxFaceIndex = 0;
        for j = 1:length(faces)
            currentBox = bboxes(j, :);
            currentArea = currentBox(3) * currentBox(4);
            if currentArea > maxFaceArea
                maxFaceArea = currentArea;
                maxFaceIndex = j;
            end
        end

        % 只处理最大的人脸
        if maxFaceIndex > 0
            face = faces{maxFaceIndex};
            bbox = bboxes(maxFaceIndex, :);

            scaleFactorX = 512 / bbox(3);
            scaleFactorY = 512 / bbox(4);
            for k = 1:4
                bbox(1 + 4 * k) = round((bbox(1 + 4 * k) - bbox(1)) * scaleFactorX);
                bbox(2 + 4 * k) = round((bbox(2 + 4 * k) - bbox(2)) * scaleFactorY);
                bbox(3 + 4 * k) = round(bbox(3 + 4 * k) * scaleFactorX);
                bbox(4 + 4 * k) = round(bbox(4 + 4 * k) * scaleFactorY);
            end

            resizedFace = imresize(face, [512, 512], 'Method', 'bilinear');
            alignedFace = alignFace(resizedFace, bbox);
            alignedFace = imresize(alignedFace, [353, 353], 'Method', 'bilinear');

            % 调用 PCA 人脸识别
            detect3(alignedFace, trainDir, detector);

            % 保存对齐图像
            [~, name] = fileparts(fileName);
            outputName = sprintf('%s.jpg', name);
            outputPath = fullfile(outputDir, outputName);
            imwrite(alignedFace, outputPath);
        end

    catch ME
        fprintf('处理文件 %s 出错: %s\n', fileName, ME.message);
    end
end

disp('处理完成！');
