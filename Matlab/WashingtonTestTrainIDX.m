files = importdata('depthfiles.txt');

n = size(files,1);
instancelist = zeros(n,1);

for i = 1:n
    test = strsplit(files{i},'/');
    test = test{8};
    test = test(end);
    instancelist(i) = str2num(test);
end

trainidx = find(instancelist~=1);
testidx = find(instancelist==1);

dlmwrite('trainidx_0.txt',trainidx(randperm(length(trainidx))));
dlmwrite('testidx_0.txt',testidx(randperm(length(testidx))));

shuffled = randperm(n);
frac = fix(n/10);

train = zeros(n-frac,10);
test = zeros(frac,10);
traincell = cell(1,10);
testcell = cell(1,10);

for i = 1:10
    start = (1 + (i-1)*frac)
    fin = i*frac
    test(:,i) = shuffled(start:fin);
    train(:,i) = shuffled(setdiff(1:n,start:fin));
    testcell{i} = setdiff(test(:,i),trainidx,'stable');
    traincell{i} = setdiff(train(:,i),testidx,'stable');
end

for i = 1:10
    str = int2str(i);
    dlmwrite(strcat('trainidx_',str,'.txt'),traincell{i});
    dlmwrite(strcat('testidx_',str,'.txt'),testcell{i});
end