function res = BASPNN_classifier(X_train, X_test, y_train)
d_lim= 0.1;
alpha1=-1;
sigma1=0.1;
% %% Read the data and normlize it within 0 to 1  
% X_train = csvread('X_train0.csv');
% y_train = csvread('y_train0.csv');
% y_train=y_train';
% % y_train=int32(y_train);
% X_test = csvread('X_test0.csv');
% y_test = csvread('y_test0.csv');
% y_test=y_test';
% 
% S
% % y_test =int32(y_test)
% disp(size(X_train)) 
% disp(size(y_train))


data = [X_train,y_train'];

% test_data =[X_test,y_test'];
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

for i=1: size(X_train(:,1))
     if(data(i,end)==0)
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
% total = size_c1 +size_c2;

prior_c1 = 1/size_c1;
prior_c2 = 1/size_c2;

w1 = train_c1(:,1:size(train_c1,2)-1);
% y1 = train_c1(:,size(train_c1,2));

w2 = train_c2(:,1:size(train_c2,2)-1);
% y2 = train_c2(:,size(train_c2,2));




%% input dataset to be tested

X = X_test;

for i=1:size(X(:,1))
    for j= 1:size(X(1,:))-1
      X(i,j)=X(i,j)/sqrt(norm(X(i,1:size(X(1,:)))));
    end
end
% given_y = test_data(:,size(test_data,2));

% [best,fmin,N_iter] = bat_algorithm(5,5,0.5,0.5,data,data,total,d_lim);
% 
% sigma1 = best(1:size_c1);
% sigma2 = best(size_c1+1:total);
dim = size(X,2);

%% New technique for laplace dataset
tic

% % % summ1 = []; summ2 = [];
% % % for i = 1:size(X,1)
% % %    for j = 1:size(w1,1)
% % % %       xx = X(i,:)-w1(j,:);
% % %         % dists = (X(i,:)-w1(j,:))*(X(i,:)-w1(j,:))';
% % % %       summ1(i,j) = (1/(sqrt(2*pi)*sigma1(i))^dim)* (vij(j) * exp(-1*max(sum(abs(xx)))/sigma1(j)));
% % %         summ1(i,j) = skewNormalKernel(w1(j,:), X(i,:), sigma1, alpha1);
% % % 
% % %    end 
% % % end    
% % % sum1 = sum(summ1,2);
% % % 
% % % 
% % % for i = 1:size(X,1)
% % %    for j = 1:size(w2,1)
% % %         % dists = (X(i,:)-w2(j,:))*(X(i,:)-w2(j,:))';
% % % %        summ2(i,j) = (1/(sqrt(2*pi)*sigma2(i))^dim)*(vij(j) * exp(-1*max(sum(abs(yy)))/sigma2(j)));
% % %         summ2(i,j) = skewNormalKernel(w2(j,:), X(i,:), sigma1, alpha1);
% % % 
% % %    end 
% % % end    
% % % sum2 = sum(summ2,2);

summ1 = []; summ2 = [];
for i = 1:size(X,1)
%       xx = X(i,:)-w1(j,:);
        % dists = (X(i,:)-w1(j,:))*(X(i,:)-w1(j,:))';
%       summ1(i,j) = (1/(sqrt(2*pi)*sigma1(i))^dim)* (vij(j) * exp(-1*max(sum(abs(xx)))/sigma1(j)));
        summ1(i) = skewNormalKernel(w1, X(i,:), sigma1, alpha1);
end    
sum1 = summ1';


for i = 1:size(X,1)

        % dists = (X(i,:)-w2(j,:))*(X(i,:)-w2(j,:))';
%        summ2(i,j) = (1/(sqrt(2*pi)*sigma2(i))^dim)*(vij(j) * exp(-1*max(sum(abs(yy)))/sigma2(j)));
        summ2(i) = skewNormalKernel(w2, X(i,:), sigma1, alpha1);
     
end    
sum2 = summ2';


value1 = prior_c1*sum1 ;
value2 = prior_c2*sum2;


computed_y = [];
for i=1:size(value1,1)
    if(value1(i) > value2(i))
        computed_y(i) = 0;      
    else        
        computed_y(i) = 1;
    end
end

res=computed_y'; 
