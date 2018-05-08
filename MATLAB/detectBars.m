v = VideoReader('greenBarsTest.avi');
load('stereoParameters.mat');
j=1;
while hasFrame(v)
    stereoimg = readFrame(v);
    
    imgLeft = stereoimg(1:540, 481:1440, :);
    imgRight = stereoimg(541:1080, 481:1440, :);
    
    imshow(imgLeft);
    %imwrite(imgRight, 'imgRight.jpg');
    imgLeftHSV = rgb2hsv(imgLeft);
    %     imshow(imgLeft);
    %     figure;
    %      imshow(imgLeftHSV(:,:,2));
    %      imgLeftA = imgLeftHSV(:,:,1);
    %       imshow(imgLeftA,[]);
    %     imgLeftSat = imgLeftLAB(:,:,3);
    %imgLeftSat = adapthisteq(imgLeftSat);
    greenMask = imgLeftHSV(:,:,1)<=0.7 & imgLeftHSV(:,:,1) >=0.2...
        & imgLeftHSV(:,:,2) >=0.1;
    se = strel('disk', 5);
    %imgLeftHue = imgaussfilt(imgLeftHue);
    %     greenMask = imgLeftHSV(:,:,2)>-45 & imgLeftHSV(:,:,2)<=0 &...
    %         imgLeftHSV(:,:,3) >=0 & imgLeftHSV(:,:,3) < 45;
    greenMask = imclose(greenMask, se);
    areaBars = regionprops(greenMask, 'Area');
    bars = imclose(bwareaopen(greenMask, 50), se);
    barEdges = edge(bars);
    imshow(barEdges);
    
    toolProps = regionprops(barEdges,'Orientation', 'PixelIdxList', 'Centroid');
    cntrMat = [];
    for i = 1: length(toolProps)
        cntrMat = [cntrMat;toolProps(i).Centroid];
    end
    toolIden = kmeans(cntrMat,2);
    
    for i= 1:2
        tools{i} = toolProps(find(toolIden ==i),:);
    end
    
    for i = 1:length(tools)
        for j = 1: length(tools{i})
            toolBarsCenter(j,:) = tools{i}(j).Centroid;
            [y, x] = ind2sub(size(barEdges), tools{i}(j).PixelIdxList);
            toolPoints{i}{j} = [x,y];
        end
        for j = 1 : length(tools{i})
            if length(toolBarsCenter) >1
                toolBarLine = [toolBarsCenter(j,:), (toolBarsCenter(2,2)- toolBarsCenter(1,2))...
                    /(toolBarsCenter(2,1)-  toolBarsCenter(1,1)) ] ;
                toolBarPts = toolPoints{i}{j};
                barcornr = getBarCorners(toolBarPts, toolBarLine);
                toolBarCorners{i}{j}= barcornr;
            end
            toolCurves{i}{j} = removeLines(toolPoints{i}{j}, toolBarCorners{i}{j});
            hold on;scatter(toolCurves{i}{j}(:,1), toolCurves{i}{j}(:,2));
        end
    end
end
