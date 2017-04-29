function [ R ] = Rodriguez( vec, phi )
% Using Rodriguez rotation formula, gets the rotation matrix about vector
% vec by angle phi.
% Inputs 
%   vec = 1x3 or 3x1 normalized vector for rotation direction
%   phi = angle in rads we wish to rotate by
% Outputs
%   R = 3x3 rotation matrix

    skew_sym = [ 0 -vec(3) vec(2); vec(3) 0 -vec(1);-vec(2) vec(1) 0];
    R = eye(3) + sin(phi)*skew_sym + (1-cos(phi))*skew_sym^2;
end

