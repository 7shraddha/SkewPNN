function y_predict = d_tree(x_train, x_test, y_train)
%D_FOREST Summary of this function goes here
%   Detailed explanation goes here
    % model = fit_Hellinger_forest(x_train, y_train,10,[],[],5);
    % y_predict = predict_Hellinger_forest(model,x_test);
    model = fit_Hellinger_tree(x_train, y_train,[],5);
    [y_predict,scores] = predict_Hellinger_tree(model,x_test);
    % [~,~,~,AUC] = perfcurve(y_test,scores,1);  
    % y_predict = [y_predict;AUC];
end

