function linesCombined = combineCloseLines(lines)

lineFlag = zeros(1,length(lines));
k=0;
for i = 1:length(lines)-1
    if lineFlag(i) == 0
        k=k+1;
        lineFlag(i) = k;
        for j = i+1:length(lines)
            if (lines(j).theta < lines(i).theta + 15) && (lines(j).theta > lines(i).theta - 15)...
                    && (lines(j).rho < lines(i).rho + 50) && (lines(j).rho > lines(i).rho - 50)
                lineFlag(j)= k;
            end
        end
    else
        continue;
    end
end

linesGrouped = struct([]);
for i=1: max(lineFlag)
    idxLine = find(lineFlag == i) ;
    for j = 1:length(idxLine)
        linesGrouped{i,j} = lines(idxLine(j));
    end
end

linesCombined = [];

for i = 1:size(linesGrouped,1)
    allPtsLine = [];
    for j = 1:length(linesGrouped(i,:))
        if isempty(linesGrouped{i,j}) == 1
            continue;
        end
        point1 = linesGrouped{i,j}.point1;
        point2 = linesGrouped{i,j}.point2;
        lengthLine = pdist([point1 ; point2], 'euclidean');
        t = 0:1/lengthLine:1;
        ptsLine = repmat(point1, length(t),1)' +(point2- point1)'*t;
        allPtsLine = [allPtsLine, ptsLine];
    end
    markerLength = pdist([max(allPtsLine'); min(allPtsLine')], 'euclidean');
    [mLine,cLine] = ransac_line(allPtsLine,2, 500, 3, 0.8);
    linePoint1 = mean(allPtsLine,2) - (markerLength/2) * [cos(atan(mLine)); sin(atan(mLine))];
    linePoint2 = mean(allPtsLine,2) + (markerLength/2) * [cos(atan(mLine)); sin(atan(mLine))];
    linesCombined = [linesCombined, [linePoint1; linePoint2;mLine]];
end
end


