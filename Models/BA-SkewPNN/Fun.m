function z=Fun(u , data, test_data)


data_1=[]; data_2 = [];


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



X = test_data(:,1:size(test_data,2)-1);
given_y = test_data(:,size(test_data,2));


summ1 = []; summ2 = [];

dim = size(X,2);

for i = 1:size(X,1)
   for j = 1:size(w1,1)

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


%%  Performing PNN






value1 = prior_c1  * sum1 ;
value2 = prior_c2  * sum2;




computed_y = [];
for i=1:size(value1,1)
    if(value1(i) > value2(i))
        computed_y(i) = 1;      
    else        
        computed_y(i) = 2;
    end
end


[conf_mat_test1,~] = confusionmat(given_y,computed_y);

accuracy_p=100 * sum(diag(conf_mat_test1))/sum(conf_mat_test1(:));
precision_p = 100 * conf_mat_test1(1,1)/(conf_mat_test1(1,1)+conf_mat_test1(2,1));
recall_p = 100 * conf_mat_test1(1,1)/(conf_mat_test1(1,1)+conf_mat_test1(1,2));
Specificity_p =  100 * conf_mat_test1(2,2)/(conf_mat_test1(2,1)+conf_mat_test1(2,2));

f1_measure_p = 2 * precision_p * recall_p /(precision_p + recall_p);
AUC = (recall_p +Specificity_p)/2;
 
z = - (accuracy_p+f1_measure_p + AUC);

end

