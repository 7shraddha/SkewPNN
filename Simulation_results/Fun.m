function z=Fun(u , data, test_data)
% Sphere function with fmin=0 at (0,0,...,0)

data_1=[]; data_2 = [];
% for i=1:size(data(:,1))
%     for j= 1:size(data(1,:))-1
%       data(i,j)=data(i,j)/sqrt(norm(data(i,1:size(data(1,:))-1)));
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


total = size_c1 + size_c2  ;

prior_c1 = 1/size_c1;
prior_c2 = 1/size_c2;



w1 = train_c1(:,1:size(train_c1,2)-1);
y1 = train_c1(:,size(train_c1,2));

w2 = train_c2(:,1:size(train_c2,2)-1);
y2 = train_c2(:,size(train_c2,2));

sigma1 = u(1:size_c1);
sigma2 = u(size_c1+1:total);


% %% Calculating weights for weighted PNN
% inter =1;
% d1=[];
% for i = 1: size(w1,1)
%     d1(i) = inter /norm(exp(-abs(mean(w1)))-exp(-abs(w1(i,:))),1);
% end
% 
% d2=[];
% for i = 1: size(w2,1)
%     d2(i) = inter /norm(exp(-abs(mean(w2)))-exp(-abs(w2(i,:))),1);
% end
% % vij = [d1 , d2];


%% input dataset to be tested

% test_data = [d_110(:,:);d_210(:,:);];
X = test_data(:,1:size(test_data,2)-1);
given_y = test_data(:,size(test_data,2));

% sigma1 = u; 

% X(1,:)
% w1(1,:)
% xx = X(1,:)-w1(1,:)
% dot(xx,xx)
% -1*dot(xx,xx)/(2*sigma^2)
% exp(-1*dot(xx,xx)/(2*sigma^2))
summ1 = []; summ2 = [];

dim = size(X,2);

for i = 1:size(X,1)
   for j = 1:size(w1,1)
%       xx = X(i,:)-w1(j,:);
        dists = (X(i,:)-w1(j,:))*(X(i,:)-w1(j,:))';
%       summ1(i,j) = (1/(sqrt(2*pi)*sigma1(i))^dim)* (vij(j) * exp(-1*max(sum(abs(xx)))/sigma1(j)));
        summ1(i,j) =(1/(sqrt(2*pi)*sigma1(j)))*(exp(-dists/(2*sigma1(j)^2)));
        
   end 
end    
sum1 = sum(summ1,2);


for i = 1:size(X,1)
   for j = 1:size(w2,1)
        dists = (X(i,:)-w2(j,:))*(X(i,:)-w2(j,:))';
%        summ2(i,j) = (1/(sqrt(2*pi)*sigma2(i))^dim)*(vij(j) * exp(-1*max(sum(abs(yy)))/sigma2(j)));
        summ2(i,j) = (1/(sqrt(2*pi)*sigma2(j)))*(exp(-dists/(2*sigma2(j))));
     
   end 
end    
sum2 = sum(summ2,2);


%%  Performing classic PNN

% temp1 = exp(-(X*w1'-1) / (2*sigma))
% size(temp1)
% sum(temp1(:,:))





value1 = prior_c1  * sum1 ;
value2 = prior_c2  * sum2;

% value1 = radbas(sum1);
% value2 = radbas(sum2);

% value1 =  exp(-(sum1)/(2*sigma));
% value2 =  exp(-(sum2)/(2*sigma));


computed_y = [];
for i=1:size(value1,1)
    if(value1(i) > value2(i))
        computed_y(i) = 1;      
    else        
        computed_y(i) = 2;
    end
end

% z = norm(given_y - computed_y',2);
[conf_mat_test1,~] = confusionmat(given_y,computed_y);

accuracy_p=100 * sum(diag(conf_mat_test1))/sum(conf_mat_test1(:));
precision_p = 100 * conf_mat_test1(1,1)/(conf_mat_test1(1,1)+conf_mat_test1(2,1));
recall_p = 100 * conf_mat_test1(1,1)/(conf_mat_test1(1,1)+conf_mat_test1(1,2));
Specificity_p =  100 * conf_mat_test1(2,2)/(conf_mat_test1(2,1)+conf_mat_test1(2,2));
% G_measure_p = sqrt(precision_p * Recall_p) 
f1_measure_p = 2 * precision_p * recall_p /(precision_p + recall_p);
AUC = (recall_p +Specificity_p)/2;
% perform = Specificity_p*log2(Recall_p) + Recall_p*log2(Specificity_p);
% accuracy_p=100 * sum(diag(conf_mat_test1))/sum(conf_mat_test1(:));
% % z = - (recall_p *log2(Specificity_p) + Specificity_p * log2(recall_p)); 
z = - (accuracy_p+f1_measure_p + AUC);
% %z=sum(u.^2);
end

