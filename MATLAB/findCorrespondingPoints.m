function [matchedLeftPoints,matchedRightPoints] = findCorrespondingPoints(leftPoints, rightPoints, fundamentalMatrix)
%FINDCORRESPONDINGPOINTS Summary of this function goes here
%   Detailed explanation goes here
for i = 1:size(leftPoints,1)
    for j = 1: size(leftPoints, 2)
        if isempty(leftPoints{i,j}) ==0 && isempty(rightPoints{i,j}) ==0
            dataLeft = leftPoints{i,j};
            dataRight = rightPoints{i,j};
            [matchedLeftPts, matchedRightPts] = isPointOnLine(dataLeft, dataRight, fundamentalMatrix);
            matchedLeftPoints{i,j} = matchedLeftPts;
            matchedRightPoints{i,j} = matchedRightPts;
            
%             subplot(1,2,1);hold on;scatter(matchedLeftPoints{i,j}(:,1), matchedLeftPoints{i,j}(:,2));
%             subplot(1,2,2);hold on;scatter(matchedRightPoints{i,j}(:,1), matchedRightPoints{i,j}(:,2));
        end
    end
end


end

