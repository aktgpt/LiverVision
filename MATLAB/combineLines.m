function linesLeftCombined = combineLines(linesLeft)

linesLeftCombined = struct([]);
i=1;
k=1;
for j=1:length(linesLeft)-1
    tempLineStruct{i}(k)= linesLeft(j);
    if linesLeft(j+1).theta == linesLeft(j).theta && linesLeft(j+1).rho == linesLeft(j).rho
        k=k+1;
    else
        k=k+1;
        i = i+1;
        k=1;
    end
end

for i=1: length(tempLineStruct)
    linesLeftCombined(i).rho = tempLineStruct{i}(1).rho;
    linesLeftCombined(i).theta = tempLineStruct{i}(1).theta;
    linesLeftCombined(i).point1 = tempLineStruct{i}(1).point1;
    linesLeftCombined(i).point2 = tempLineStruct{i}(length(tempLineStruct{i})).point2;
end


