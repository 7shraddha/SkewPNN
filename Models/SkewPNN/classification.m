function [res]=classification(data,test_data,sigma1,alpha1)

% For two-class datasets

data=data(randperm(size(data,1)),:);
test_data=test_data(randperm(size(test_data,1)),:);

data_1 = [] ; data_2=[];

for i=1: size(data(:,1))
     if(data(i,end)==1)
         data_1= [data_1;data(i,:)];
     else
         data_2= [data_2;data(i,:)];
     end    
end

train_c1 = data_1;
train_c2 = data_2;

% finding the priori probability  
size_c1 = size(train_c1,1);
size_c2 = size(train_c2,1);


prior_c1 = 1/size_c1;
prior_c2 = 1/size_c2;

w1 = train_c1(:,1:size(train_c1,2)-1);



w2 = train_c2(:,1:size(train_c2,2)-1);



%% input dataset to be tested

X = test_data(:,1:size(test_data,2)-1);
given_y = test_data(:,size(test_data,2));

dim = size(X,2);

% PNN parameters
% Kernel width parameter
% Skewness parameter





summ1 = []; summ2 = [];

for i = 1:size(X,1)
   for j = 1:size(w1,1)

      summ1(i,j) = skewNormalKernel(w1(j,:), X(i,:), sigma1, alpha1);
   end 
end    
sum1 = sum(summ1,2);

for i = 1:size(X,1)
   for j = 1:size(w2,1)

       summ2(i,j) = skewNormalKernel(w2(j,:), X(i,:), sigma1, alpha1);
   end 
end    
sum2 = sum(summ2,2);



value1 = prior_c1   * sum1 ;
value2 = prior_c2   * sum2;


computed_y = [];
for i=1:size(value1,1)
    if(value1(i) > value2(i))
        computed_y(i) = 1;      
    else        
        computed_y(i) = 2;
    end
end



[conf_mat_test6,~] = confusionmat(given_y,computed_y);
accuracy_6=100 * sum(diag(conf_mat_test6))/sum(conf_mat_test6(:));
precision_6 = 100 * conf_mat_test6(1,1)/(conf_mat_test6(1,1)+conf_mat_test6(2,1));
recall_6 = 100 * conf_mat_test6(1,1)/(conf_mat_test6(1,1)+conf_mat_test6(1,2));
Specificity_6 =  100 * conf_mat_test6(2,2)/(conf_mat_test6(2,1)+conf_mat_test6(2,2));
f1_measure_6 = 2 * precision_6 * recall_6 /(precision_6 + recall_6);

% Calculate ROC curve and AUC
AUC = (recall_6 +Specificity_6)/2;


res = [ sigma1,alpha1, accuracy_6 , precision_6 , recall_6 ,Specificity_6, f1_measure_6,AUC ];

writematrix (res, 'res_10fold_haberman.csv', 'WriteMode', 'append');


end
% END