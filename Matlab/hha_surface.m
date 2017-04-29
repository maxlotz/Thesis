clearvars -except depths rawDepths

for i=1:1449
    str = int2str(i);
    % SURFACE NORMALS + HHA ENCODING
    % Adapted from Gupta's github page
    dims = [480 640 3];

    % Set max height of images in dataset for normalisation
    max_height = 5.73;

    % Minimum distance in m, so that disparity does not become too large.
    % Should correspond to closest distance object can be to camera.
    mindist = 0.3;
    maxdist = 10;

    test_depth = rawDepths(:,:,i);
    pcl = depth_plane2depth_world(test_depth);

    % Guptas method of surface normal estimation, he uses clipping radius = 10
    % for estimation of gravity vector and clipping radius = 3 for surface
    % normals, However examination of the angle image when using radius = 3
    % shows it is very noisy, so 10 has been used (more blurry though)
    [surfnorm_grav, ~] = computeNormalsSquareSupport(pcl*100,test_depth==0,10,1,ones(size(test_depth)));
    %[surfnorm_pcl, ~] = computeNormalsSquareSupport(pcl*100,test_depth==0,3,1,ones(size(test_depth)));
    surfnorm_pcl = surfnorm_grav;

    % % Matlabs surface normal estimator
    % ptCloud = pointCloud(pcl);
    % normals = pcnormals(ptCloud,10);

    % Get estimated gravity normal using Guptas method
    yDir = estimated_gravity(reshape(surfnorm_grav,[numel(surfnorm_grav)/3,3]));

    % Get angle between Y axis and estimated gravity
    phi = acos(yDir*[0 1 0]');

    % Get normalised vector perpendicular to Y axis and estimated gravity
    % vector in which we wish to rotate pointcloud
    vec = cross([0 1 0],yDir);
    vec = vec/norm(vec);

    % Calculate rotation matrix using Rodriguez rotation formula
    R = Rodriguez(vec,phi);

    % Rotate pointcloud by rotation matrix (equivalent to pc = (R'*pcl')'
    pclrot = pcl*R;

    % HORIZONTAL DISPARITY
    % Truncate values below minimum distance (should be none)
    pcl(:,3) = min(max(pcl(:,3), mindist),maxdist);
    disparity = 1./pcl(:,3);
    disparity = 255*(disparity - 1/maxdist)/(1/mindist - 1/maxdist);
    disparity = reshape(disparity,dims(1:end-1));

    % HEIGHT
    % Remember pixels increase downwards.
    height = -pclrot(:,2);
    % Lowest height assumed to be ground
    height = height - min(height(:));
    % Truncates values above max height (should be none)
    height = min(height, max_height);
    % Normalises height
    height = height/max_height*255;
    height = reshape(height,dims(1:end-1));

    % ANGLE
    angle = angle_norms(reshape(surfnorm_pcl,[numel(surfnorm_pcl)/3,3]),yDir);
    angle = reshape(angle,dims(1:end-1));
    angle = angle*255/180;

    HHA = cat(3, disparity, height, angle);
    HHA = uint8(HHA);
    surfnorm = uint8(surfnorm_grav*255);
    
    imwrite(HHA,strcat('NYUV2_raw_HHA_', str,'.png'));
    imwrite(surfnorm,strcat('NYUV2_raw_surfnorm_', str,'.png'));
end