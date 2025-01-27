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


    gray_hist=imhist(sample_train,256);

    gray_hist=gray_hist/double(max(sample_train(:)));
    features_train=[features_train;gray_hist'];

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
     gray_hist=imhist(sample_train,256);

    gray_hist=gray_hist/double(max(sample_train(:)));
    features_train=[features_train;gray_hist'];
    labels_train=[labels_train;1];
% features_train=[features_train;sample_train(:)'];
end
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
   gray_hist=imhist(sample_test,256);

    gray_hist=gray_hist/double(max(sample_test(:)));
    features_test=[features_test;gray_hist'];
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
    gray_hist=imhist(sample_test,256);

    gray_hist=gray_hist/double(max(sample_test(:)));
    features_test=[features_test;gray_hist'];
    labels_test=[labels_test;1];
    % features_test=[features_test;sample_test(:)'];
end
%% knn classifer
acc=[];
for k=1:1:100
predict_label=0;
for i=1:1:size(features_test,1)
    predict_label(i)=knn_classifier(features_train,labels_train,features_test(i,:),k);
end
accuracy=100*sum(predict_label'==labels_test)/length(labels_test);
acc=[acc;accuracy];
end
% k =31 acc 61%