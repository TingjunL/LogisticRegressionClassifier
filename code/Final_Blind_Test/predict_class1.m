% here is the final logistic regression classifier that we built, the
% discriminator, input dataset and the prior of the classifier can be
% defined by users. After call this function, users can gain the predicted results. 

function [data_predicted] = predict_class(Discriminator,original_data,prior)

data = original_data(:,1:end)';
[m,n] = size(data);
data = [data;ones(1,n)];
%% Predict class
dist = Discriminator*data;%/norm(Discriminator);
predicted_class = 1./(1+exp(-dist));
% figure(1);
% hist(predicted_class,50);

%% Determine Threshold

threshold = 0;
totalsize = length(predicted_class);
while true
    pos = sum(predicted_class<threshold);
    ratio = pos/totalsize;
    if ratio > prior
        break;
    else
        threshold = threshold + 0.005;
    end
end

pos_index = find(predicted_class<threshold);
neg_index = find(predicted_class>=threshold);
data = data(1:end-1,:);
data(end,pos_index)= 1;
data(end,neg_index) = 0;

data_predicted = data';
% data_predicted = [data,original_data(:,end)];





end