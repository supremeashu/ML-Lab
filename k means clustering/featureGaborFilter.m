clc;
clear all;
f_path1="C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\train\dogs";
f_path2="C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\train\cats";
file1=dir(fullfile(f_path1,"","*.jpg"));
file2=dir(fullfile(f_path2,"","*.jpg"));
total_images=length(file1)+length(file2);

features_train=[];
labels_train=[];
%% loading images of dog and getting its training vector
for i=1:1:length(file1)
    img_path=fullfile(file1(i).folder,file1(i).name);
    sample_train=imread(img_path);
    % sample_train=imresize(sample_train,[32,32]);
    if(size(sample_train,3)<3)
        sample_train=imresize(sample_train,[32,32]);
    else
        sample_train=imresize(rgb2gray(sample_train),[32,32]);
    end
    

    g=gabor([2,4],[0,45,90,135]);
    gaborimage=imgaborfilt(sample_train,g);
    features=[];
    for j=1:1:length(g)
        filtered_image=gaborimage(:,:,j);
        features=[features,mean(filtered_image(:)),std(filtered_image(:))];
    end

    features_train=[features_train;features];
    labels_train=[labels_train;0];
end
%% loading images of cat and getting its tarining vector
for i=1:1:length(file2)
    img_path=fullfile(file2(i).folder,file2(i).name);
    sample_train=imread(img_path);
    % sample_train=imresize(sample_train,[32,32]);
    if(size(sample_train,3)<3)
        sample_train=imresize(sample_train,[32,32]);
    else
        sample_train=imresize(rgb2gray(sample_train),[32,32]);
    end
    g=gabor([2,4],[0,45,90,135]);
    gaborimage=imgaborfilt(sample_train,g);
    features=[];
    for j=1:1:length(g)
        filtered_image=gaborimage(:,:,j);
        features=[features,mean(filtered_image(:)),std(filtered_image(:))];
    end

    features_train=[features_train;features];
    labels_train=[labels_train;1];
% features_train=[features_train;sample_train(:)'];
end
%% Run k-means on training features
k = 2; % Number of clusters
max_iter = 100; % Maximum iterations for k-means
[cluster_labels_train, centroids] = kmeans_func(features_train, k, max_iter);

% Relabel clusters to match actual labels
% Find the majority label in each cluster
label_map = zeros(k, 1); % Mapping from cluster index to actual label
for j = 1:k
    labels_in_cluster = labels_train(cluster_labels_train == j);
    if ~isempty(labels_in_cluster)
        label_map(j) = mode(labels_in_cluster); % Most common label in the cluster
    end
end

% Map cluster labels to actual labels
mapped_labels_train = label_map(cluster_labels_train);

% Calculate training accuracy
training_accuracy = 100 * sum(mapped_labels_train == labels_train) / length(labels_train);
%% Testing here
f_path3="C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\test\dogs";
f_path4="C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\test\cats";
file3=dir(fullfile(f_path3,"","*.jpg"));
file4=dir(fullfile(f_path4,"","*.jpg"));
total_images_1=length(file3)+length(file4);
features_test=[];
labels_test=[];

%% loading dog test image and extracting test vectors
for i=1:length(file3)
    img_path=fullfile(file3(i).folder,file3(i).name);
    sample_test=imread(img_path);
    if(size(sample_test,3)<3)
        sample_test=imresize(sample_test,[32,32]);
    else
        sample_test=imresize(rgb2gray(sample_test),[32,32]);
    end
    g=gabor([2,4],[0,45,90,135]);
    gaborimage=imgaborfilt(sample_test,g);
    features=[];
    for j=1:1:length(g)
        filtered_image=gaborimage(:,:,j);
        features=[features,mean(filtered_image(:)),std(filtered_image(:))];
    end

    features_test=[features_test;features];
    labels_test=[labels_test;0];
    % features_test=[features_test;sample_test(:)'];
end
%% loading images of cat and getting its testing vector
for i=1:1:length(file4)
    img_path=fullfile(file4(i).folder,file4(i).name);
    sample_test=imread(img_path);
    if(size(sample_test,3)<3)
        sample_test=imresize(sample_test,[32,32]);
    else
        sample_test=imresize(rgb2gray(sample_test),[32,32]);
    end
    g=gabor([2,4],[0,45,90,135]);
    gaborimage=imgaborfilt(sample_test,g);
    features=[];
    for j=1:1:length(g)
        filtered_image=gaborimage(:,:,j);
        features=[features,mean(filtered_image(:)),std(filtered_image(:))];
    end

    features_test=[features_test;features];
    labels_test=[labels_test;1];
    % features_test=[features_test;sample_test(:)'];
end
%% k-means classifer

% Assign test samples to clusters based on nearest centroid
predicted_labels_test = zeros(size(labels_test));
for i = 1:size(features_test, 1)
    distances = sum((centroids - features_test(i, :)).^2, 2); % Squared Euclidean distance
    [~, cluster_idx] = min(distances); % Find the closest centroid
    predicted_labels_test(i) = label_map(cluster_idx); % Map cluster index to actual label
end

% Calculate testing accuracy
test_accuracy = 100 * sum(predicted_labels_test == labels_test) / length(labels_test)

% acc = 55%