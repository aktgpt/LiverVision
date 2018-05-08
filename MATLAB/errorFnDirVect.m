function errorFn = errorFnDirVect(dirVect)
%ERRORFNDIRVECT Summary of this function goes here
%   Detailed explanation goes here
global pt;
errorFn = mean(distPointsfromLine(pt, dirVect, mean(pt)));


end

