function worldPoints = triangulatePoints(matchedLeftPoints,matchedRightPoints,stereoParams)
%TRIANGULATEPOINTS Summary of this function goes here
%   Detailed explanation goes here
params = load('stereoParameters.mat');

global ptsCurve;
global meanPtCurve;
global plane;
worldPtCloud = [];
worldPtCloudTool = cell(2,1);
% initlineVect = [1,1,1,1,0,0];

for i = 1:size(matchedLeftPoints,1)
    worldPtCloudShiftedTool{i} = [];
    k = 1;
    for j = 1: size(matchedLeftPoints, 2)
        if length(matchedLeftPoints{i,j}) > 10
            worldPoints{i,j} = triangulate(matchedLeftPoints{i,j}, matchedRightPoints{i,j}, stereoParams);
            meanWorldPoints{i,j} = mean(worldPoints{i,j});
            worldPtCloudTool{i} = [worldPtCloudTool{i}; worldPoints{i,j}];
            worldPtCloud = [worldPtCloud; worldPoints{i,j}];
            
%             hold on; pcshow(pointCloud(worldPoints{i,j}));
%             hold on; plot3(0,0,0,'Marker','o');
            %curves planes
            ptsCurve = worldPoints{i,j};
            meanPtCurve = mean(ptsCurve);
            [n, V, p] = affine_fit(ptsCurve);
            d = -dot(n,p);
            plane= planeModel([n', d]);
            planeCurves{i,j} =plane;
%             hold on;  plot(planeCurves{i,j});
            
            %finding point on plane
            initPtOnPlane = meanPtCurve;
            optimizedPtOnPlane{i,j} = fmincon(@errorFnPlane,initPtOnPlane,[],[],[],[],[],[],@constraintPtOnPlane);
%             hold on; scatter3( optimizedPtOnPlane{i,j}(1), optimizedPtOnPlane{i,j}(2), optimizedPtOnPlane{i,j}(3));
            
            %reproject point back to images
            toolPointsLeft{i}(k,:) = worldToImage(params.stereoParams.CameraParameters1,...
                eye(3),[0,0,0], optimizedPtOnPlane{i,j});
            toolPointsRight{i}(k,:) = worldToImage(params.stereoParams.CameraParameters2,...
                params.stereoParams.RotationOfCamera2,params.stereoParams.TranslationOfCamera2, optimizedPtOnPlane{i,j});
            k = k+1;
        end
    end
    
    %     tool center line
    xLeft = [ones(length(toolPointsLeft{i}),1), toolPointsLeft{i}(:,1)];
    yLeft = toolPointsLeft{i}(:,2);
    bLeft = xLeft\ yLeft;
    yLeftCalc = xLeft* bLeft;
        subplot(1,2,1);hold on;scatter(xLeft(:,2), yLeft,'b');
        subplot(1,2,1);hold on;plot(xLeft(:,2), yLeftCalc,'b');
    
    xRight = [ones(length(toolPointsRight{i}),1), toolPointsRight{i}(:,1)];
    yRight = toolPointsRight{i}(:,2);
    bRight = xRight \ yRight;
    yRightCalc = xRight*bRight;
        subplot(1,2,2);hold on;scatter(xRight(:,2), yRight,'b');
        subplot(1,2,2);hold on;plot(xRight(:,2), yRightCalc,'b');
    
    %     initLinePt = mean(worldPtCloudTool{i});
    %     initDirVect = fitNormal(worldPtCloudTool{i})';
    %     initLine = [initDirVect ,initLinePt];
    %     %     pt = worldPtCloudTool{i};
    %     %     hold on; pcshow(pointCloud(worldPtCloudTool{i}));
    %     %     hold on; plot3(0,0,0,'Marker','o');
    %     %     plane = pcfitplane(pointCloud(pt),1);
    %     %     hold on; plot(plane);
    %     optimizedline{i} = fmincon(@errorFnLine,initLine,[],[],[],[],[],[],@constraintLine);
end


% hold on; showExtrinsics(params.stereoParams );

% figure;pcshow(pointCloud(worldPtCloud));
% hold on;quiver3(optimizedline{2}(4),optimizedline{2}(5),optimizedline{2}(6)...
%     , optimizedline{2}(1),optimizedline{2}(2),optimizedline{2}(3), 'Color','r');
%
% coeff = [];
% for j = 1:length(worldPoints)
%     if isempty(worldPoints{1,j}) ==0
%         pts = worldPoints{1,j};
%         meanPts(j,:) = mean(pts);
%         coeff{j} = pca(pts);
%         hold on;quiver3(meanPts(j,1),meanPts(j,2),meanPts(j,3), coeff{j}(1,2),coeff{j}(1,1),coeff{j}(1,3), 'Color','r');
%         hold on;quiver3(meanPts(j,1),meanPts(j,2),meanPts(j,3), coeff{j}(2,2),coeff{j}(2,1),coeff{j}(2,3), 'Color','g');
%         hold on;quiver3(meanPts(j,1),meanPts(j,2),meanPts(j,3), coeff{j}(3,2),coeff{j}(3,1),coeff{j}(3,3), 'Color','b');
%     else
%         continue;
%     end
% end
% pts = worldPtCloudTool{1};
% %%point projection on plane
% for i= 1:length(pts)
%     projectedpts(i,:) = projPointOnPlane(pts(i,:), coeff{1}(2,:), coeff{1}(3,:) );
%     projectedpts(i,:) = projectedpts(i,:)/ projectedpts(i,3);
% end
%
% hold on;scatter(projectedpts(:,1),projectedpts(:,2),projectedpts(:,3));
%
%
%
% cylinFit = pcfitcylinder(pointCloud(worldPtCloudTool{2}), 0.05, optimizedDirVect, 10);
% hold on; plot(cylinFit);
%
% for i = 1:size(meanWorldPoints,1)
%     ptOnLine = [optimizedLine{i}(1),optimizedLine{i}(2),optimizedLine{i}(3)];
%     lineVect = [optimizedLine{i}(4),optimizedLine{i}(5),optimizedLine{i}(6)];
%     for j = 1: size(meanWorldPoints, 2)
%         if isempty(meanWorldPoints{i,j}) == 0
%             pointsOnTool{i}(j,:) = ptOnLine - dot(meanWorldPoints{i,j}-ptOnLine, lineVect)*lineVect;
%             %hold on; scatter(pointsOnTool{i,j}(1),pointsOnTool{i,j}(2),pointsOnTool{i,j}(3));
%         else
%             pointsOnTool{i}(j,:)= [NaN,NaN,NaN];
%             continue;
%         end
%     end
%     hold on; line(pointsOnTool{i}(:,1), pointsOnTool{i}(:,2), pointsOnTool{i}(:,3));
% end

% hold on; plot3(0,0,0,'Marker','o');
end

