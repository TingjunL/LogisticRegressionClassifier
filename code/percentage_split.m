% This function is derived for split the data set that we got. Most of the
% time we just use weka to split the data directly.


function [training_data,test_data]= percentage_split(dataset)
number_instance=size(dataset,1);
number_feature=size(dataset,2);
nn=1;
mm=1;
%%seperate positive data and negative data into 2 datasets
for i=1:number_instance
    if dataset(i,number_feature)==1
        data_positive(nn,:)=dataset(i,:);
        nn=nn+1;
    else
        data_negative(mm,:)=dataset(i,:);
        mm=mm+1;
    end
end



%%calculate number
number_positive=size(data_positive,1);
number_negative=size(data_negative,1);

%%creat new id
for i=1:number_positive
    id_positive(i)=i;
end

for i=1:number_negative
    id_negative(i)=i;
end
id_positive=id_positive';
id_negative=id_negative';
data_positive=[id_positive data_positive];
data_negative=[id_negative data_negative];

ratio_p=number_positive/number_instance;

number_70=fix(number_instance*0.7);
number_30=number_instance-number_70;

number_70_positive=fix(number_70*ratio_p);
number_70_negative=number_70-number_70_positive;
number_30_positive=fix(number_30*ratio_p);
number_30_negative=number_30-number_30_positive;
%%generate all p numbers random value
p_series=randperm(number_positive);
p_70=p_series(1:number_70_positive);
p_30=p_series(number_70_positive+1:number_positive);
n_series=randperm(number_negative);
n_70=n_series(1:number_70_negative);
n_30=n_series(number_70_negative+1:number_negative);


for i=1:number_30_positive
    split_30_positive(i,:)=data_positive(find(data_positive(:,1)==p_30(i)),:);
    data_positive(find(data_positive(:,1)==p_30(i)),:)=[];
end

split_70_positive=data_positive;
for i=1:number_30_negative-1
    split_30_negative(i,:)=data_negative(find(data_negative(:,1)==n_30(i)),:);
    data_negative(find(data_negative(:,1)==n_30(i)),:)=[];
end

split_70_negative=data_negative;

training_data=cat(1,split_70_positive,split_70_negative);
test_data=cat(1,split_30_positive,split_30_negative);
training_data(:,1)=[];
test_data(:,1)=[];
end