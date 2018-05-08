function errorFn = errorFnPlane(ptOnPlane)
%ERRORFNPLANE Summary of this function goes here
%   Detailed explanation goes here

global ptsCurve; 
for i = 1:length(ptsCurve)
distFn(i)= abs(pdist([ptsCurve(i,:);ptOnPlane])- 2.5 );
end
errorFn = mean(distFn);
end

