function [linept1, linept2] = lineProject3D(line1,line2, stereoParams)
%LINEPROJECT3D Summary of this function goes here
%   Detailed explanation goes here

alpha = 30;
line1pt1 = [line1.center(1) + (alpha*cos(line1.orientation)), line1.center(2) + (alpha*sin(line1.orientation)), 1];
line1pt2 = [line1.center(1) - (alpha*cos(line1.orientation)), line1.center(2) - (alpha*sin(line1.orientation)), 1];

line2pt1 = [line2.center(1) + (alpha*cos(line2.orientation)), line2.center(2) + (alpha*sin(line2.orientation)), 1];
line2pt2 = [line2.center(1) - (alpha*cos(line2.orientation)), line2.center(2) - (alpha*sin(line2.orientation)), 1];

line1Homo = cross(line1pt1, line1pt2);
line2Homo = cross(line2pt1, line2pt2);

p1 = stereoParams.CameraParameters1.IntrinsicMatrix'*[eye(3), [0; 0; 0]];

p2 = stereoParams.CameraParameters2.IntrinsicMatrix'*[stereoParams.RotationOfCamera2', stereoParams.TranslationOfCamera2'];

p1inv = p1' * inv(p1*p1');
H = p2 * p1inv;

line3D = [line1Homo*p1; line2Homo*p2];

linpt3D = null(line3D);
linpt3D = linpt3D';

lambda = 3000;
mu = -3000;

linept1 = lambda*linpt3D(1,:) + mu*linpt3D(2,:);
linept1 = linept1/linept1(4);

linept2 = lambda*linpt3D(1,:) - mu*linpt3D(2,:);
linept2 = linept2/linept2(4);

end

