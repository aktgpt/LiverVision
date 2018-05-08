function [c,ceq] = constraintPtOnPlane(ptOnPlane)
%CONSTRAINTPTONPLANE Summary of this function goes here
%   Detailed explanation goes here
global meanPtCurve;
global plane;
c = norm(meanPtCurve) - norm(ptOnPlane);
ceq = dot(plane.Parameters, [ptOnPlane,1]);

end

