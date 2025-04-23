function alignedFace = alignFace(face, bbox)
% 提取关键点坐标
% bbox
% leftEye = [bbox(5) + bbox(7) / 2, bbox(6) + bbox(8) / 2];
% rightEye = [bbox(9) + bbox(11) / 2, bbox(10) + bbox(12) / 2];
% mouth = [bbox(13) + bbox(15) / 2, bbox(14) + bbox(16) / 2];
% nose = [bbox(17) + bbox(19) / 2, bbox(18) + bbox(20) / 2];

% 构建源点关键点矩阵
% landmarks = [
%     leftEye;
%     rightEye;
%     mouth;
%     nose;
%     ];

% landmarks

% landmarks = bbox(1:4); % Updated to use bbox directly
% landmarks
% imshow(face);

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

landmarks = []; % 初始化为空数组
targetPoints = []; % 初始化目标点数组

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

% landmarks


% 在原图上标记关键点并显示
% figure;
% imshow(face);
% hold on;
% plot(landmarks(:, 1), landmarks(:, 2), 'r*', 'MarkerSize', 10); % 添加关键点标记
% title('人脸关键点标记');
% hold off;

% 计算仿射变换矩阵
transformationMatrix = fitgeotrans(landmarks, targetPoints, 'projective');
% transformationMatrix = fitgeotrans(landmarks, targetPoints, 'affine');

% 应用仿射变换
outputView = imref2d(size(face));
alignedFace = imwarp(face, transformationMatrix, 'OutputView', outputView);

end