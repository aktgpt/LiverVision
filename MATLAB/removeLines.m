function [curvePts, linePts] = removeLines(barContour,barCorners)
%REMOVELINES Summary of this function goes here
%   Detailed explanation goes here

numPts = size(barContour', 2);
threshDist =5;
inlierIdx = [];
curvePts = barContour;
for i = 1:2
    %distance of lines
    point1 = barCorners(2*i-1, :)';
    point2 = barCorners(2*i, :)';
    kLine = point2-point1;% two points relative distance
    kLineNorm = kLine/norm(kLine);
    normVector = [-kLineNorm(2),kLineNorm(1)];%Ax+By+C=0 A=-kLineNorm(2),B=kLineNorm(1)
    distance = normVector*(barContour' - repmat(point1,1,numPts));
    
    inlierIdx = [inlierIdx,find(abs(distance)<=threshDist)] ; 
    inlierMat = barContour(inlierIdx,:);
    
%     for j = 1:length(inlierMat)
%         binImg(inlierMat(2,j), inlierMat(1,j)) =0;
%     end
end
linePts = curvePts(inlierIdx,:);
curvePts(inlierIdx,:) = [];

end
% binOpImg = binImg;
% imshow(binImg);
% end

