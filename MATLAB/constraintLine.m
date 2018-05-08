function [c,ceq] = constraintLine(lineVar)
%CONSTRAINTLINE Summary of this function goes here
%   Detailed explanation goes here
global plane;
c = [];
ceq = [sign(dot([lineVar(1), lineVar(2),lineVar(3), 1], plane.Parameters)) + sign(plane.Parameters(4));...
    norm([lineVar(1), lineVar(2),lineVar(3)]) - 1];

end

