% Here we find out the data from unlabelled training data which we beilive
% can help us optimaized the classifier. After the choice, we can get the
% full imformation of these datas.

clear;
clc;
addpath('../confusion_table');

%% Determine chosen data
load('../Discriminator.mat');
data = load('../UnlabelledTrain_Selected_replace.csv');


data = data';
[m,n] = size(data);
data = [data;ones(1,n)];

%% Find decision boundary
threshold = 0.3;
totalsize = n;
dist = Discriminator_Enchance'*data;
predicted_class = 1./(1+exp(-dist));
figure(1);
hist(predicted_class,100);
while true
    pos = sum(predicted_class<threshold);
    ratio = pos/totalsize;
    if ratio > 1/7
        break;
    else
        threshold = threshold + 0.001;
    end
end

left = 0.781;
right = 0.787;
[index_boundary] = find_attributes(Discriminator_Enchance',data,left,right);

% index_boundary = index_boundary';

% csvwrite('index.csv',index_boundary);
% index_boundary = index_boundary(1:100);
unknown_data = data(:,index_boundary);
dist_unknown = Discriminator_Enchance'*unknown_data;
predicted_class_unknown = 1./(1+exp(-dist_unknown));

temp = data(1:end-1,index_boundary)';
temp = [index_boundary',temp];
csvwrite('100boundary_labelled3.csv',temp);















