function res = PNN_classifier(X_train, X_test, y_train)
sigma1=0.1;



data = [X_train,y_train'];

data_1 = [] ; data_2=[];

for i=1:size(data(:,1))
    for j= 1:size(data(1,:))-1
      data(i,j)=data(i,j)/sqrt(norm(data(i,1:size(data(1,:))-1)));
    end
end



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
total = size_c1 +size_c2;

prior_c1 = 1/size_c1;
prior_c2 = 1/size_c2;

w1 = train_c1(:,1:size(train_c1,2)-1);
y1 = train_c1(:,size(train_c1,2));

w2 = train_c2(:,1:size(train_c2,2)-1);
y2 = train_c2(:,size(train_c2,2));

%% input dataset to be tested

X = X_test;

for i=1:size(X(:,1))
    for j= 1:size(X(1,:))-1
      X(i,j)=X(i,j)/sqrt(norm(X(i,1:size(X(1,:)))));
    end
end

tic



%disp("performing calculations in pattern layers, label 0")
summ1 = []; summ2 = [];
con = (1/(sqrt(2*pi)*sigma1));
con2 = 1/(2*sigma1^2);
for i = 1:size(X,1)
    dist = sum((X(i,:)-w1).^2,2);
    summ1(i) = sum(con*(exp(-dist*con2)));
end
sum1 = summ1';

%disp("performing calculations in pattern layers, label 1")
for i = 1:size(X,1)
    dist = sum((X(i,:)-w2).^2,2);
    summ2(i) = sum(con*(exp(-dist*con2)));
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
% disp('end');
res=computed_y'; 
