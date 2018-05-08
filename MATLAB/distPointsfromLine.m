function distMat = distPointsfromLine(ptMat,dirVect, linePt)
%DISTPOINTSFROMLINE Summary of this function goes here


for i = 1:length(ptMat)
ptsVect = ptMat(i,:) - linePt;
distMat(i) = norm(norm(cross(ptsVect, dirVect))/norm(dirVect));
end

end

