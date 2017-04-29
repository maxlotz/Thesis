clearvars -except files
depthdims = [256 256];
rgbdims = [256 256 3];
imsize = [431 347];
maxd = 10.15; % max depth of 10m

files = importdata('depthfiles.txt');
datasize = size(files,1);

for i=1:datasize
    
    test_depth = imread(files{i});
    loc = load(findloc(files{i}));
    rgb = imread(findrgb(files{i}));
    [pcl depth] = depthToCloud(test_depth,loc);
    depth = depth - min(depth(:));
    depth = depth/max(depth(:));

    orig_size = size(depth);
    [maxsize idx] = max(orig_size);
    stretch = 256/maxsize;
    stretch_size = round(orig_size*stretch);
    rgb = imresize(rgb,stretch_size);
    depth = imresize(depth,stretch_size);

    rgbnew = zeros(rgbdims);
    depthnew = zeros(depthdims);

    lower = 128 - round(min(stretch_size)/2);
    upper = lower + min(stretch_size);

    if idx == 2
        rgbnew(lower+1:upper,:,:) = rgb;
        depthnew(lower+1:upper,:) = depth;
    end

    if idx == 1
        rgbnew(:,lower+1:upper,:) = rgb;
        depthnew(:,lower+1:upper) = depth;
    end


    str = int2str(i);
    imwrite(uint8(depthnew*255),parula(256),strcat('Washington_raw_parula_stretch_norm_', str,'.png'));
    imwrite(uint8(depthnew*255),hsv(256),strcat('Washington_raw_hsv_stretch_norm_', str,'.png'));
    imwrite(uint8(depthnew*255),jet(256),strcat('Washington_raw_jet_stretch_norm_', str,'.png'));
    imwrite(uint8(depthnew*255),strcat('Washington_raw_mono_stretch_norm_', str,'.png'))

    i
    
end