function [res]=classification(data,test_data,d_lim,alpha1)

%% Read the data and normlize it within 0 to 1  

% %% Seperate training and testing dataset  
% 
data_1 = [] ; data_2=[];

for i=1:size(data(:,1))
    for j= 1:size(data(1,:))-1
      data(i,j)=data(i,j)/sqrt(norm(data(i,1:size(data(1,:))-1)));
    end
end

% data= data./repmat(sqrt(norm(data(:,1:size(data(1,:))-1))));
% for i=1:size(test_data(:,1))
%     for j= 1:size(test_data(1,:))-1
%       test_data(i,j)=test_data(i,j)/sqrt(norm(test_data(i,1:size(test_data(1,:))-1)));
%     end
% end

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
total = size_c1 +size_c2;

prior_c1 = 1/size_c1;
prior_c2 = 1/size_c2;

w1 = train_c1(:,1:size(train_c1,2)-1);
y1 = train_c1(:,size(train_c1,2));

w2 = train_c2(:,1:size(train_c2,2)-1);
y2 = train_c2(:,size(train_c2,2));


%% input dataset to be tested

X = test_data(:,1:size(test_data,2)-1);
given_y = test_data(:,size(test_data,2));

[best,fmin,N_iter] = bat_algorithm(5,5,0.5,0.5,data,data,total,d_lim);
    
sigma1 = best(1:size_c1);
sigma2 = best(size_c1+1:total);
dim = size(X,2);

%% New technique for laplace dataset
tic
summm1 = []; summ2 = [];
for i = 1:size(X,1)
   for j = 1:size(w1,1)
%       
        dists = (X(i,:)-w1(j,:))*(X(i,:)-w1(j,:))';

        summ1(i,j) = skewNormalKernel(w1(j,:), X(i,:), sigma1(j), alpha1);
        
   end 
end    
sum1 = sum(summ1,2);


for i = 1:size(X,1)
   for j = 1:size(w2,1)
        dists = (X(i,:)-w2(j,:))*(X(i,:)-w2(j,:))';

        summ2(i,j) = skewNormalKernel(w2(j,:), X(i,:), sigma2(j), alpha1);
     
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

 etime_6 = toc;  
[conf_mat_test6,~] = confusionmat(given_y,computed_y);
accuracy_6=100 * sum(diag(conf_mat_test6))/sum(conf_mat_test6(:));
precision_6 = 100 * conf_mat_test6(1,1)/(conf_mat_test6(1,1)+conf_mat_test6(2,1));
recall_6 = 100 * conf_mat_test6(1,1)/(conf_mat_test6(1,1)+conf_mat_test6(1,2));
Specificity_6 =  100 * conf_mat_test6(2,2)/(conf_mat_test6(2,1)+conf_mat_test6(2,2));

f1_measure_6 = 2 * precision_6 * recall_6 /(precision_6 + recall_6);

AUC = (recall_6 +Specificity_6)/2;

res = [ etime_6,alpha1, accuracy_6 , precision_6 , recall_6 ,Specificity_6, f1_measure_6,AUC ];

writematrix (res, 'res_10fold_haberman.csv', 'WriteMode', 'append');

end
% END