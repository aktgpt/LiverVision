function [matchedLeftPoints, matchedRightPoints] = isPointOnLine(dataPointsRight, dataPointsLeft, fundamentalMatrix)
% Is point Q=[x3,y3] on line through P1=[x1,y1] and P2=[x2,y2]
% Normal along the line:
dataPointsRight(:,3) = 1;
dataPointsLeft(:,3) = 1;

for i = 1 : length(dataPointsLeft)
    lineRight = fundamentalMatrix * dataPointsLeft(i,:)';  
    for j = 1 : length(dataPointsRight)
        distMat(j) =  dataPointsRight(j,:) * lineRight;
    end
    distMat = abs(distMat);
    distMat(distMat > 1e-2) = 1;
    [minDist, idx] = min(abs(distMat));
    minDistMat(i,:) = [minDist, idx];
end


for i = 1: length(dataPointsRight)
    potentialRightPts = find(minDistMat(:,2) ==i);
    if isempty(potentialRightPts) ==1
        matchedLeftPtsIdx(i) = 0;
    else
        if length(potentialRightPts) == 1
            matchedLeftPtsIdx(i) = potentialRightPts;
        else
            distMat = minDistMat(:, 1);
            distCompMat = distMat(potentialRightPts);
            [minptDist,corresPt] = min(distCompMat);
            if minptDist ~= 1
                matchedLeftPtsIdx(i) = potentialRightPts(corresPt);
            else
                matchedLeftPtsIdx(i) = 0; 
            end
        end
    end
end

matchedLeftPoints = dataPointsRight(find(matchedLeftPtsIdx),:);
matchedLeftPoints(:,3) = [];
matchedRightPoints = dataPointsLeft(matchedLeftPtsIdx(find(matchedLeftPtsIdx)), :);
matchedRightPoints(:,3) = [];

end
