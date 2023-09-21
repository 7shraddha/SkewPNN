% PNN program
% Probablistic Neural Network
% Programmer Shraddha Naik

%% 
clc;
clear all;
close all;

res=[];res_fs=[]; 


[ train_fold_1, train_fold_2, train_fold_3, train_fold_4, train_fold_5, train_fold_6, train_fold_7, ...
    train_fold_8, train_fold_9, train_fold_10, test_fold_1, test_fold_2, test_fold_3, test_fold_4, ...
    test_fold_5, test_fold_6, test_fold_7, test_fold_8, test_fold_9, test_fold_10 ] = code_10_fold();


sigma1=0;
alpha1 = [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6];

for j=1:size(alpha1,2)
    for i=1:20
        sigma1 = sigma1 + 0.01;   
    r1_fs = classification(train_fold_1,test_fold_1,sigma1,alpha1(j));
    r2_fs = classification(train_fold_2,test_fold_2,sigma1,alpha1(j));
    r3_fs = classification(train_fold_3,test_fold_3,sigma1,alpha1(j));
    r4_fs = classification(train_fold_4,test_fold_4,sigma1,alpha1(j));
    r5_fs = classification(train_fold_5,test_fold_5,sigma1,alpha1(j));
    r6_fs = classification(train_fold_6,test_fold_6,sigma1,alpha1(j));
    r7_fs = classification(train_fold_7,test_fold_7,sigma1,alpha1(j));
    r8_fs = classification(train_fold_8,test_fold_8,sigma1,alpha1(j));
    r9_fs = classification(train_fold_9,test_fold_9,sigma1,alpha1(j));
    r10_fs = classification(train_fold_10,test_fold_10,sigma1,alpha1(j));
    res_total=[r1_fs;r2_fs;r3_fs;r4_fs;r5_fs;r6_fs;r7_fs;r8_fs;r9_fs;r10_fs];
    res_mean =  mean(res_total);
    % writematrix (res_mean, 'Results_haberman.csv','WriteMode', 'append');
    end
end
