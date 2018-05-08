v = VideoReader('greenBarsTest.avi');
load('stereoParameters.mat');
j=1;
p1 = stereoParams.CameraParameters1.IntrinsicMatrix'*[eye(3), [0; 0; 0]];

p2 = stereoParams.CameraParameters2.IntrinsicMatrix'*[stereoParams.RotationOfCamera2', stereoParams.TranslationOfCamera2'];

K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
K2 = stereoParams.CameraParameters2.IntrinsicMatrix';
R = stereoParams.RotationOfCamera2;
T = stereoParams.TranslationOfCamera2;

H = K1*R*inv(K2);
e01 = K1 * (T');

p1inv = p1' * inv(p1*p1');
H1 = p2 * p1inv;

F = stereoParams.FundamentalMatrix;

% [dummy, dummy, v] = svd(F);
% ep1 = v(:,3)/v(3,3);
%
% [dummy, dummy, v] = svd(F');
% ep2 = v(:,3)/v(3,3);
frameCount = 0;
while hasFrame(v)
    stereoimg = readFrame(v);
    
    imgLeft = stereoimg(1:540, 481:1440, :);
    imgRight = stereoimg(541:1080, 481:1440, :);
    
        subplot(1,2,1);imshow(imgLeft);
        subplot(1,2,2);imshow(imgRight);
    
    barsLeft = barEdges(imgLeft);
    barsRight = barEdges(imgRight);
    
%     subplot(1,2,1);imshow(barsLeft);
%     subplot(1,2,2);imshow(barsRight);
    
    toolsLeft = groupTools(barsLeft);
    toolsRight = groupTools(barsRight);
    
    imgSize = size(imgLeft);
    
    [toolsCurveLeft, toolsLinesLeft] = getToolCurves(toolsLeft, imgSize);
    [toolsCurveRight, toolsLinesRight] = getToolCurves(toolsRight, imgSize);
    
    %     for i = 1 : size(toolsCurveLeft,1)
    %         for j = 1 : size(toolsCurveLeft,2)
    %             if isempty(toolsCurveLeft{i,j}) ==0 && isempty(toolsCurveRight{i,j}) ==0
    %                 subplot(1,2,1);hold on;scatter(toolsCurveLeft{i,j}(:,1), toolsCurveLeft{i,j}(:,2));
    %                 subplot(1,2,2);hold on;scatter(toolsCurveRight{i,j}(:,1), toolsCurveRight{i,j}(:,2));
    %             end
    %         end
    %     end
    
    
%     for i = 1 : size(toolsLinesLeft,1)
%         for j = 1 : size(toolsLinesLeft,2)
%             if isempty(toolsLinesLeft{i,j}) ==0
%                 subplot(1,2,1);hold on;scatter(toolsCurveLeft{i,j}(:,1), toolsCurveLeft{i,j}(:,2));
%                 subplot(1,2,2);hold on;scatter(toolsCurveRight{i,j}(:,1), toolsCurveRight{i,j}(:,2));
%                 subplot(1,2,1);hold on;scatter(toolsLinesLeft{i,j}(:,1), toolsLinesLeft{i,j}(:,2));
%                 subplot(1,2,2);hold on;scatter(toolsLinesRight{i,j}(:,1), toolsLinesRight{i,j}(:,2));
%             end
%         end
%     end
    
    [matchedLeftCurves,matchedRightCurves] = findCorrespondingPoints(toolsCurveLeft, toolsCurveRight, F);
    [matchedLeftLines,matchedRightLines] = findCorrespondingPoints(toolsLinesLeft, toolsLinesRight, F);
    
%     for i = 1 : size(matchedLeftCurves,1)
%         for j = 1 : size(matchedLeftCurves,2)
%             if isempty(matchedLeftCurves{i,j}) ==0 && isempty(matchedRightCurves{i,j}) ==0
% %                 subplot(1,2,1);hold on;scatter(matchedLeftLines{i,j}(:,1), matchedLeftLines{i,j}(:,2));
% %                 subplot(1,2,2);hold on;scatter(matchedRightLines{i,j}(:,1), matchedRightLines{i,j}(:,2));
%                 subplot(1,2,1);hold on;scatter(matchedLeftCurves{i,j}(:,1), matchedLeftCurves{i,j}(:,2));
%                 subplot(1,2,2);hold on;scatter(matchedRightCurves{i,j}(:,1), matchedRightCurves{i,j}(:,2));
%             end
%         end
%     end
    
    
     worldPointsCurves = triangulatePoints(matchedLeftCurves,matchedRightCurves,stereoParams);
%     worldPointsLines = triangulatePoints(matchedLeftLines,matchedRightLines,stereoParams);

    
    frameCount = frameCount+1;
end