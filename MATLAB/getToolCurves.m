function [toolsCurveSeparated, toolsLinesSeparated] = getToolCurves(tools, imgSize)
%GETTOOLCURVES Summary of this function goes here
%   Detailed explanation goes here
% toolPoints = cell(size(tools));
% toolBarCorners = cell(size(tools));
% toolCurves = cell(size(tools));

for i = 1:length(tools)
    for j = 1: length(tools{i})
        toolBarsCenter(j,:) = tools{i}(j).Centroid;
        [y, x] = ind2sub(imgSize, tools{i}(j).PixelIdxList);
        toolPoints{i, j} = [x,y];
    end
    
    
    for j = 1 : length(tools{i})
        if length(toolBarsCenter) > 1
            toolBarLine = [toolBarsCenter(j,:), (toolBarsCenter(2,2) - toolBarsCenter(1,2))...
                /(toolBarsCenter(2,1)-  toolBarsCenter(1,1))] ;
            toolBarPts = toolPoints{i,j};
            barcornr = getBarCorners(toolBarPts, toolBarLine);
            toolBarCorners{i,j} = barcornr;
        end
         [curvePts, linePts]= removeLines(toolPoints{i,j}, toolBarCorners{i,j});
         toolCurves{i,j} = curvePts;
         toolLines{i,j} = linePts;
%         hold on;scatter(toolCurves{i,j}(:,1), toolCurves{i,j}(:,2));
    end
end

toolsCurveSeparated = cell(size(toolCurves,1), 2*size(toolCurves,2));

for i = 1:size(toolCurves,1)
    for j = 1: size(toolCurves,2)
        if isempty(toolCurves{i,j}) == 0
            data = toolCurves{i,j};
            idx = kmeans(data,2);
            idxUpdate = zeros(size(idx));
            if idx(1) ~= 1
                idxUpdate(idx ==1) = 2 ;
                idxUpdate(idx ==2) = 1;
            else
                idxUpdate = idx;
            end
            toolsCurveSeparated{i,2*j-1} = data(idxUpdate ==1,:);
            toolsCurveSeparated{i,2*j} = data(idxUpdate ==2,:);
        end
    end
end


toolsLinesSeparated = cell(size(toolLines,1), 2*size(toolLines,2));

for i = 1:size(toolLines,1)
    for j = 1: size(toolLines,2)
        if isempty(toolLines{i,j}) == 0
            data = toolLines{i,j};
            idx = kmeans(data,2);
            idxUpdate = zeros(size(idx));
            if idx(1) ~= 1
                idxUpdate(idx ==1) = 2 ;
                idxUpdate(idx ==2) = 1;
            else
                idxUpdate = idx;
            end
            toolsLinesSeparated{i,2*j-1} = data(idxUpdate ==1,:);
            toolsLinesSeparated{i,2*j} = data(idxUpdate ==2,:);
        end
    end
end



end

