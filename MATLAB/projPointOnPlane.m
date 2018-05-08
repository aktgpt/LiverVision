function projectedPoint = projPointOnPlane(pt,baseVect1, baseVect2)
%PROJPOINTONPLANE Summary of this function goes here
%   Detailed explanation goes here

projectedPoint = dot(pt, baseVect1)*baseVect1 + dot(pt, baseVect2)*baseVect2;

end

