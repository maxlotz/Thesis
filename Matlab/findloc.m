function [ loc ] = findloc( path )
%Given full path of Washington RGBD depth file, returns file name of loc
%file
% Inputs:
%   path    full file path of depth image, type char
% Outputs:
%   loc     full file path of loc image, type char

    path = strsplit(path,'.');
    path = path{1};
    path = path(1:end-9);
    loc = strcat(path,'loc.txt');

end

