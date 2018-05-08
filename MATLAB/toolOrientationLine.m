load('stereoParameters.mat');

v = VideoReader('2018.03.06 11.39.06.mp4');

j=1;
while hasFrame(v)  
    stereoimg = readFrame(v);
    
    imgLeft = stereoimg(1:540, 481:1440, :);
    imgRight = stereoimg(541:1080, 481:1440, :);
    [frameLeftRect, frameRightRect] = ...
        rectifyStereoImages(imgLeft, imgRight, stereoParams);
    
    hsvImageLeft = rgb2hsv(frameLeftRect);
    hsvImageRight = rgb2hsv(frameRightRect);
    
    satImageLeft = hsvImageLeft(:,:,2);
    satImageRight = hsvImageRight(:,:,2);
    
    se = strel('disk', 7);
    binImageLeft = imcomplement(imclose(imbinarize(satImageLeft, 0.4), se));
    binImageRight = imcomplement(imclose(imbinarize(satImageRight, 0.4), se));
    
    topbboxLeft = bwareafilt(binImageLeft, 2);
    topbboxRight = bwareafilt(binImageRight, 2);
    
    leftToolscentroid = regionprops(topbboxLeft, 'centroid');
    rightToolscentroid = regionprops(topbboxRight, 'centroid');
    
    leftToolsOrientation = regionprops(topbboxLeft, 'Orientation');
    rightToolsOrientation = regionprops(topbboxRight, 'Orientation');
    
    for i= 1:length(leftToolscentroid)
        lineleft.center = leftToolscentroid(i).Centroid;
        lineleft.orientation = leftToolsOrientation(i).Orientation;
        
        lineright.center = rightToolscentroid(i).Centroid;
        lineright.orientation = rightToolsOrientation(i).Orientation;
        
        [toolpt1, toolpt2] = lineProject3D(lineleft, lineright, stereoParams);
        toolCord{j} = [toolpt1(:,1:3); toolpt2(:,1:3)];
        j=j+1;
    end
    %     f = figure;
    %     subplot(1,2,1); imshow(topbboxLeft);
    %     subplot(1,2,2); imshow(topbboxRight);
    %     pause(0.2);
    %     close(f);
end

for i= 1:length(toolCord)
    if mod(i,2)==0
        line(toolCord{i}(:,1), toolCord{i}(:,2), toolCord{i}(:,3), 'Color', 'r');
        hold on;
    else
        line(toolCord{i}(:,1), toolCord{i}(:,2), toolCord{i}(:,3), 'Color', 'g');
        hold on;
    end
end