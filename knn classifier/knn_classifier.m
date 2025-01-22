function [predict_label] = knn_classifier(features_train, labels_train, features_test,k)

dist=0;

for i = 1:1:size(features_train,1)
    
    dist(i)= sqrt(sum((features_train(i,:) - features_test).^2));
end

[p,q] = sort(dist);

k_nearest_indices = q(1:k);

k_nearest_labels = labels_train(k_nearest_indices);

predict_label = mode(k_nearest_labels);