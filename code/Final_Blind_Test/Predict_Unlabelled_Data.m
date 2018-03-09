% Here we use the classifier that we choose to predict the unlabelled data.
% The results will be compare to the practical data to give the final
% presision and recall, the results shows in the project report.
% this is the final step of our project.
clear;
clc
close all;
addpath('../confusion_table');

prior = 4088/(24671+4088);

original_data = load('blind_merged.csv'); % Here is the blind data that we got
data_length = length(original_data);


%% 
load('../Discriminator_Final.mat'); % Here we load the discriminator of the classifier that we choose
[data_predicted] = predict_class1(Discriminator_Enchance',original_data,prior);
pred_class = data_predicted(:,end);


csvwrite('Tingjun_Ye_Prediced_Class_Logistic_Regression.csv',pred_class);

histogram(pred_class);