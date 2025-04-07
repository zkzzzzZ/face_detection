function [bbox, bbX, faces] = detectFaceParts(detector, X, thick)
    % 简化版检测函数（已移除不必要的绘制代码）

    if nargin < 3
        thick = -1; % 默认不绘制
    end

    % 检测人脸
    bbox = step(detector.detector{5}, X);
    bbsize = size(bbox);
    partsNum = zeros(size(bbox, 1), 1);

    % 检测面部器官
    stdsize = detector.stdsize;

    for k = 1:4

        if (k == 1)
            region = [1, int32(stdsize * 2/3); 1, int32(stdsize * 2/3)];
        elseif (k == 2)
            region = [int32(stdsize / 3), stdsize; 1, int32(stdsize * 2/3)];
        elseif (k == 3)
            region = [1, stdsize; int32(stdsize / 3), stdsize];
        elseif (k == 4)
            region = [int32(stdsize / 5), int32(stdsize * 4/5); int32(stdsize / 3), stdsize];
        else
            region = [1, stdsize; 1, stdsize];
        end

        bb = zeros(bbsize);

        for i = 1:size(bbox, 1)
            XX = X(bbox(i, 2):bbox(i, 2) + bbox(i, 4) - 1, bbox(i, 1):bbox(i, 1) + bbox(i, 3) - 1, :);
            XX = imresize(XX, [stdsize, stdsize]);
            XX = XX(region(2, 1):region(2, 2), region(1, 1):region(1, 2), :);

            b = step(detector.detector{k}, XX);

            if (size(b, 1) > 0)
                partsNum(i) = partsNum(i) + 1;

                if (k == 1)
                    b = sortrows(b, 1);
                elseif (k == 2)
                    b = flipud(sortrows(b, 1));
                elseif (k == 3)
                    b = flipud(sortrows(b, 2));
                elseif (k == 4)
                    b = flipud(sortrows(b, 3));
                end

                ratio = double(bbox(i, 3)) / double(stdsize);
                b(1, 1) = int32((b(1, 1) - 1 + region(1, 1) - 1) * ratio + 0.5) + bbox(i, 1);
                b(1, 2) = int32((b(1, 2) - 1 + region(2, 1) - 1) * ratio + 0.5) + bbox(i, 2);
                b(1, 3) = int32(b(1, 3) * ratio + 0.5);
                b(1, 4) = int32(b(1, 4) * ratio + 0.5);

                bb(i, :) = b(1, :);
            end

        end

        bbox = [bbox, bb];

        p = (sum(bb') == 0);
        bb(p, :) = [];
    end

    % 后处理
    bbox = [bbox, partsNum];
    bbox(partsNum <= 2, :) = [];

    % 准备输出
    bbX = X; % 原始图像，不绘制方框
    faces = cell(size(bbox, 1), 1);

    for i = 1:size(bbox, 1)
        y1 = bbox(i, 2);
        y2 = bbox(i, 2) + bbox(i, 4) - 1;
        x1 = bbox(i, 1);
        x2 = bbox(i, 1) + bbox(i, 3) - 1;
        faces{i} = X(y1:y2, x1:x2, :);
    end

end
