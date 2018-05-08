function corners = getBarCorners(data,line)
%GETBARCORNERS Summary of this function goes here
%   Detailed explanation goes here
mxLine = line(3);
origin = line(1:2);

% rotate around origin
theta = -atan(mxLine);
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];

tlIdx = 1; trIdx = 1; blIdx = 1; brIdx = 1;
rotatedData = zeros(size(data));
for i = 1:length(data)
    rotatedData(i,:) = R*(data(i,:)' - origin');
    if rotatedData(i,1) > 0 && rotatedData(i,2) > 0
        if brIdx == 1
            brIdx = i;
        end
        if norm(rotatedData(i,:)) >= norm(rotatedData(brIdx,:))
            brIdx= i;
        end
    end
    if rotatedData(i,1) < 0 && rotatedData(i,2) > 0
        if blIdx == 1
            blIdx = i;
        end
        if norm(rotatedData(i,:)) >= norm(rotatedData(blIdx,:))
            blIdx= i;
        end
    end
    if rotatedData(i,1) < 0 && rotatedData(i,2) < 0
        if tlIdx == 1
            tlIdx = i;
        end
        if norm(rotatedData(i,:)) >= norm(rotatedData(tlIdx,:))
            tlIdx= i;
        end
    end
    if rotatedData(i,1) > 0 && rotatedData(i,2) < 0
        if trIdx == 1
            trIdx = i;
        end
        if norm(rotatedData(i,:)) >= norm(rotatedData(trIdx,:))
            trIdx= i;
        end
    end
end
corners = data([trIdx;tlIdx;blIdx;brIdx],:);
% scatter(rotatedData(:,1),rotatedData(:,2));
% hold on; scatter(data(:,1),data(:,2));
%  hold on;scatter(corners(:,1), corners(:,2));
end

