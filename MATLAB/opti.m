function [x,fval,exitflag,output,lambda,grad,hessian] = opti(x0)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = optimoptions('fmincon');
%% Modify options setting
options = optimoptions(options,'Display', 'off');
[x,fval,exitflag,output,lambda,grad,hessian] = ...
fmincon(@errorFunction,x0,[],[],[],[],[],[],@constraint,options);
