radius = 2.5;
% first circle
ptCloud = cell([1,4]);
l = 1;
hold on; plot3(0,0,0,'Marker','o');
for j = -5:5:10
    k = 1;
    for i = 1:30
        x = radius*cos(i*(-360 + 720*(rand))*pi/180);
        y = radius*sin(i*(-360 + 720*(rand))*pi/180);
        z = j + (0.1+ 0.2*(rand));
        ptCloud{l}(k,:) = [x, y, z];
        k = k+1;
    end
    hold on;pcshow(pointCloud(ptCloud{l}));
    l= l+1;
end
global ptsCurve;
global plane;
for i = 1:4
ptsCurve = ptCloud{i};
meanPtCurve = mean(ptsCurve);
[n, V, p] = affine_fit(ptsCurve);
d = -dot(n,p);
plane= planeModel([n', d]);
planeCurves{i} =plane;
hold on;  plot(planeCurves{i});
%finding point on plane
initPtOnPlane = meanPtCurve;
optimizedPtOnPlane{i} = fmincon(@errorFnPlane,initPtOnPlane,[],[],[],[],[],[],@constraintPtOnPlane);
hold on; scatter3( optimizedPtOnPlane{i}(1), optimizedPtOnPlane{i}(2), optimizedPtOnPlane{i}(3));
end










% affineTform = affine3d([[rotationVectorToMatrix(optimizedline{1}(1:3));optimizedlinePt],[0;0;0;1]]);
% 
% tform = pcregrigid(pointCloud(ptCloud), pointCloud(worldPtCloudTool{1, 1}),'InitialTransform', affineTform);
% 
% pcCloudOut = pctransform(pointCloud(ptCloud), tform);
% hold on;pcshow(pcCloudOut);