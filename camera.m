vid = videoinput('winvideo', 1, 'MJPG_640x480');
vid.FramesPerTrigger = Inf;
vid.TriggerRepeat = Inf;

% 开始捕获
start(vid);

hFig = figure;
faceDetector = buildDetector();
hImage = imshow(zeros(480, 640, 3, 'uint8'));

try

    while ishandle(hFig)
        mImageCurrent = getdata(vid, 1);

        [bbox, bbimg, faces, bbfaces] = detectFaceParts(faceDetector, mImageCurrent, 1);
        set(hImage, 'CData', bbimg);
        drawnow nocallbacks;

        flushdata(vid);
    end

catch
    disp("quit")
end

% 清理资源
stop(vid);
delete(vid);
clear vid;
