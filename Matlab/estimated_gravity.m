function [ grav ] = estimated_gravity(normals)
% Estimates the angle of the gravity vector of an image, given the
% surface normals of the pixels. Method used is Gupta.
% Inputs
%   normals = nx3 matrix of X Y Z points
% Outputs
%   grav = 1x3 matrix of gravity vector

    [grav, grav0] = deal([0 1 0]);
    d = 45;

    for i = 1:10
        angle = angle_norms(normals, grav);

        if i > 5
            d = 15;
        end

        idx_par = (angle < d) | (angle > 180 - d);
        idx_perp = (angle > 90 - d) & (angle < 90 + d);

        opt = normals(idx_perp,:)'*normals(idx_perp,:) - normals(idx_par,:)'*normals(idx_par,:);
        [V,D] = eig(opt);
        [~, idx_maxeig] = min(diag(D));
        grav = V(:,idx_maxeig)';
    end
    grav = grav*sign(grav*grav0');
end