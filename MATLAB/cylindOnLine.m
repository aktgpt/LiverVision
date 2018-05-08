for i = 1:size(meanWorldPoints,1)
    for j = 1 : size(meanWorldPoints,2)
        if isempty(meanWorldPoints{i,j}) == 0
            ptsOnLine{i}(j,:) = optimizedline{i}(1,4:6) + ...
                dot(meanWorldPoints{i,j}-optimizedline{i}(1,4:6), optimizedline{i}(1,1:3))*optimizedline{i}(1,1:3);
        else
            continue;
        end
    end
    hold on;line(ptsOnLine{i}(:,1),ptsOnLine{i}(:,2),ptsOnLine{i}(:,3));
    tool{i} =cylinderModel([ptsOnLine{i}(1,:),ptsOnLine{i}(end,:),5]);
    hold on; plot(tool{i});
    
%     toolPointsLeft = worldToImage(params.stereoParams.CameraParameters1, eye(3),[0,0,0], ptsOnLine{i});
%     toolPointsRight = worldToImage(params.stereoParams.CameraParameters2,...
%         params.stereoParams.RotationOfCamera2,params.stereoParams.TranslationOfCamera2, ptsOnLine{i});
%     
%      subplot(1,2,2);hold on;scatter(toolPointsLeft(:,1), toolPointsLeft(:,2));
%      subplot(1,2,1);hold on;scatter(toolPointsRight(:,1), toolPointsRight(:,2));
end

% project points back to the image

