function imgEdges = barEdges(imageRGB)
%BAREDGES Summary of this function goes here
%   Detailed explanation goes here
imgHSV = rgb2hsv(imageRGB);

greenMask = imgHSV(:,:,1)<=0.7 & imgHSV(:,:,1) >=0.2...
    & imgHSV(:,:,2) >=0.1;
se = strel('disk', 5);

greenMask = imclose(greenMask, se);
bars = imclose(bwareaopen(greenMask, 100), se);
imgEdges = edge(bars);
end

