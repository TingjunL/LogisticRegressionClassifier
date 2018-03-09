% Generate the precision and recall of under-sample method, under sample with bagging
% method and Under-sample, Bagging and active method.
% Also find the mean and standard deviation of the precision and recall.
clear;
clc
close all;
addpath('confusion_table');

prior = 4088/(24671+4088);
original_data = load('test_data.csv');
% original_data = load('1000boundary_labelled.csv');
no_sims = 10000;
percentage = 0.3;
data_length = length(original_data);


%% Precision Recall (do nothing 1)
load('Discriminator.mat');  %Use one of the discriminator that generated from weka
Disc1 = Discriminator(:,5);
[data_predicted] = predict_class(Disc1',original_data,prior); %Use "predict_clas" to build the classifier and do classification.
pred_class = data_predicted(:,end-1);

prec_vec0 = zeros(no_sims,1);
recall_vec0 = zeros(no_sims,1);
for idx_sim = 1:no_sims
    idx = randperm(data_length,round(percentage*data_length));
    cur_data = original_data(idx,:);
    [data_predicted] = predict_class(Disc1',cur_data,prior);
    pred_class = data_predicted(:,end-1);
    actual_class = data_predicted(:,end);
    [TP,TN,FP,FN] = build_Confusion_Table(pred_class,actual_class);
    [~,recall_vec0(idx_sim),~,prec_vec0(idx_sim)] = chi_square_Contingency_table(TP,FP,FN,TN);
end

stats_recall0 = mle(recall_vec0)
stats_precision0 = mle(prec_vec0)


%% Precision Recall (do nothing 3)
Disc2 = Discriminator_Enchance; %After Bagging
[data_predicted] = predict_class(Disc2',original_data,prior);
pred_class = data_predicted(:,end-1);
actual_class = data_predicted(:,end);
[TP,TN,FP,FN] = build_Confusion_Table(pred_class,actual_class);
[~,recall_vec,~,prec_vec] = chi_square_Contingency_table(TP,FP,FN,TN)

prec_vec1 = zeros(no_sims,1);
recall_vec1 = zeros(no_sims,1);
for idx_sim = 1:no_sims
    idx = randperm(data_length,round(percentage*data_length));
    cur_data = original_data(idx,:);
    [data_predicted] = predict_class(Disc2',cur_data,1/3);
    pred_class = data_predicted(:,end-1);
    actual_class = data_predicted(:,end);
    [TP,TN,FP,FN] = build_Confusion_Table(pred_class,actual_class);
    [~,recall_vec1(idx_sim),~,prec_vec1(idx_sim)] = chi_square_Contingency_table(TP,FP,FN,TN);
end

stats_recall2 = mle(recall_vec1)
stats_precision2 = mle(prec_vec1)

%% Precision Recall (do nothing 1)

load('Discriminator_Active.mat'); %Use the discriminator that generated from weka after Active
Disc1 = Discriminator(:,9);
[data_predicted] = predict_class(Disc1',original_data,prior);
pred_class = data_predicted(:,end-1);

prec_vec0 = zeros(no_sims,1);
recall_vec0 = zeros(no_sims,1);
for idx_sim = 1:no_sims
    idx = randperm(data_length,round(percentage*data_length));
    cur_data = original_data(idx,:);
    [data_predicted] = predict_class(Disc1',cur_data,prior);
    pred_class = data_predicted(:,end-1);
    actual_class = data_predicted(:,end);
    [TP,TN,FP,FN] = build_Confusion_Table(pred_class,actual_class);
    [~,recall_vec0(idx_sim),~,prec_vec0(idx_sim)] = chi_square_Contingency_table(TP,FP,FN,TN);
end

stats_recall0 = mle(recall_vec0)
stats_precision0 = mle(prec_vec0)


%% Precision Recall (do nothing 3)
Disc2 = Discriminator_Enchance; %Active and Bagging
[data_predicted] = predict_class(Disc2',original_data,prior);
pred_class = data_predicted(:,end-1);


prec_vec1 = zeros(no_sims,1);
recall_vec1 = zeros(no_sims,1);
for idx_sim = 1:no_sims
    idx = randperm(data_length,round(percentage*data_length));
    cur_data = original_data(idx,:);
    [data_predicted] = predict_class(Disc2',cur_data,prior);
    pred_class = data_predicted(:,end-1);
    actual_class = data_predicted(:,end);
    [TP,TN,FP,FN] = build_Confusion_Table(pred_class,actual_class);
    [~,recall_vec1(idx_sim),~,prec_vec1(idx_sim)] = chi_square_Contingency_table(TP,FP,FN,TN);
end

stats_recall2 = mle(recall_vec1)
stats_precision2 = mle(prec_vec1)