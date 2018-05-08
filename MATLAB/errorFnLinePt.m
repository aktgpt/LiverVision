function errorFn = errorFnLinePt(linePt)
%ERRORFUNCTION Summary of this function goes here
%   Detailed explanation goes here

global pt; 
global optimizedDirVect;
errorFn = mean(abs(distPointsfromLine(pt, optimizedDirVect, linePt)-5));

end

% pcshow(pointCloud(pts));
% x1 = [-6.745,-4.448,12.313];
% x2 = [-4.995,-2.7480,2.613];
% hold on;line([-6.745,-4.995], [-4.448,-2.7480], [12.313,2.613]);