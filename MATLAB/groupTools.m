function tools = groupTools(edgeImg)

toolsProps = regionprops(edgeImg,'Orientation', 'PixelIdxList', 'Centroid');
cntrMat = zeros(length(toolsProps), 2);
for i = 1: length(toolsProps)
    cntrMat(i,:) = toolsProps(i).Centroid;
end
toolIden = kmeans(cntrMat,2);
toolIdenUpdate = zeros(size(toolIden));

if toolIden(1) ~= 1
    toolIdenUpdate(toolIden ==1) = 2 ;
    toolIdenUpdate(toolIden ==2) = 1;
else 
    toolIdenUpdate = toolIden;
end

for i= 1:2
    tools{i} = toolsProps(find(toolIdenUpdate ==i),:);
end
end


