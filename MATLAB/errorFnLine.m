function errorFn = errorFnLine(lineVar)
%ERRORFNLINE Summary of this function goes here
%   Detailed explanation goes here
global pt; 
% dirVect = [d1,d2,d3];
% linePt = [x,y,z];
errorFn = mean(abs(distPointsfromLine(pt, [lineVar(1), lineVar(2),lineVar(3)], [lineVar(4), lineVar(5),lineVar(6)])-5));

end

