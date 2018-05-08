function filterData=ellipseDataFilter_RANSAC(data)
% Do ellipse scatter data filtering for ellipse fitting by RANSAC method.
% Author: Zhenyu Yuan
% Date: 2016/7/26
% Ref:  http://www.cnblogs.com/yingying0907/archive/2012/10/13/2722149.html
%       Extract RANSAC filtering in ellipsefit.m and make some modification
%%  ?????
nSampLen = 3;               %??????????
nDataLen = size(data, 1);   %????
nIter = 50;                 %??????
dThreshold = 2;             %????
nMaxInlyerCount=-1;         %????
A=zeros([2 1]);
B=zeros([2 1]);
P=zeros([2 1]);
%%  ???
for i = 1:nIter 
    ind = ceil(nDataLen .* rand(1, nSampLen)); %?????nSampLen?????
    %%  ???????????????,??????????
    %???????????????????
    A(:,1)=data(ind(1),:);    %??
    B(:,1)=data(ind(2),:);    %??
    P(:,1)=data(ind(3),:);    %?????
    DIST=sqrt((P(1,1)-A(1,1)).^2+(P(2,1)-A(2,1)).^2)+sqrt((P(1,1)-B(1,1)).^2+(P(2,1)-B(2,1)).^2);
    xx=[];
    nCurInlyerCount=0;        %??????0?
    %%  ???????
    for k=1:nDataLen
        %         CurModel=[A(1,1)   A(2,1)  B(1,1)  B(2,1)  DIST ];
        pdist=sqrt((data(k,1)-A(1,1)).^2+(data(k,2)-A(2,1)).^2)+sqrt((data(k,1)-B(1,1)).^2+(data(k,2)-B(2,1)).^2);
        CurMask =(abs(DIST-pdist)< dThreshold);     %???????????????,???1
        nCurInlyerCount =nCurInlyerCount+CurMask;             %?????????????
        if(CurMask==1)
            xx =[xx;data(k,:)];
        end
    end
    %% ??????
    if nCurInlyerCount > nMaxInlyerCount   %??????????????????
        nMaxInlyerCount = nCurInlyerCount;
        %             Ellipse_mask = CurMask;
        %              Ellipse_model = CurModel;
        %              Ellipse_points = [A B P];
        filterData =xx;
    end
end