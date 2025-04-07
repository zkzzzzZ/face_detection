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

    % 处理每张图片；】E
    for i = 1:length(fileList)
        fileName = fileList(i).name;
        filePath = fullfile(inputDir, fileName);

        try
            % 读取图像并检测人脸
            img = imread(filePath);
            [~, ~, faces] = detectFaceParts(detector, img, -1); % thick=-1跳过绘制

            if isempty(faces)
                % fprintf('未检测到人脸: %s\n', fileName);
                continue;
            end

            % 处理每个检测到的人脸
            for j = 1:length(faces)
                face = faces{j};

                % 转为灰度
                if size(face, 3) == 3
                    grayFace = rgb2gray(face);
                else
                    grayFace = face; % 已经是灰度图像
                end

                grayFace = imresize(grayFace, [353, 353]);

                % 生成输出文件名
                [~, name] = fileparts(fileName);
                if j > 1
                    outputName = sprintf('%s-%d.jpg', name, j);
                else
                    outputName = sprintf('%s.jpg', name);
                end
                outputPath = fullfile(outputDir, outputName);

                % 保存图像
                imwrite(grayFace, outputPath);
            end

        catch ME
            fprintf('处理文件 %s 出错: %s\n', fileName, ME.message);
        end

    end

    disp('处理完成！');
end
