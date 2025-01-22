clc
clear all

folder_path1 = 'C:\Users\heman\Desktop\Dog-Cat Dataset\train\dogs';
folder_path2 = 'C:\Users\heman\Desktop\Dog-Cat Dataset\train\cats';

files1 = dir(fullfile(folder_path1, '**', '*.jpg'));
files2 = dir(fullfile(folder_path2, '**', '*.jpg'));

total_images = length(files1)+length(files2);

features_train = [];
labels_train = [];
%% Loading training images of dog and constructing training feature vector and training labels
for i = 1:1:length(files1)
    
       image_path = fullfile(files1(i).folder, files1(i).name);
       sample_train = imread(image_path);        
       % RGB to Gray
       if(size(sample_train,3)<3)  
       sample_train = imresize(sample_train,[32 32]);
       else
       sample_train = imresize(rgb2gray(sample_train),[32 32]);
       end
       
       labels_train = [labels_train;0]; 
       %h = hist(double(sample_train(:)'),256);h = h/length(sample_train(:));%Histogram
       features_train = [features_train;sample_train(:)'];
end
%% Loading training images of cat and constructing training feature vector and training labels
for j = 1:length(files2)
       image_path = fullfile(files2(j).folder, files2(j).name);
       sample_train = rgb2gray(imread(image_path)); 
       % RGB to Gray 
       if(size(sample_train,3)<3)  
       sample_train = imresize(sample_train,[32 32]);
       else
       sample_train = imresize(rgb2gray(sample_train),[32 32]);
       end
       
       labels_train = [labels_train;1];   
       %h =
       %hist(double(sample_train(:)'),256);h=h/length(sample_train(:));%%Histogram
       features_train = [features_train;sample_train(:)'];
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
folder_path3 = 'C:\Users\heman\Desktop\Dog-Cat Dataset\test\dogs';
folder_path4 = 'C:\Users\heman\Desktop\Dog-Cat Dataset\test\cats';

files3 = dir(fullfile(folder_path3, '**', '*.jpg'));
files4 = dir(fullfile(folder_path4, '**', '*.jpg'));

total_images = length(files3)+length(files4);

features_test = [];
labels_test = [];
%% Loading test images of dog and constructing training feature vector and training labels
for i = 1:length(files3)
    
       image_path = fullfile(files3(i).folder, files3(i).name);
       sample_test = imread(image_path);        
       % RGB to Gray
       if(size(sample_test,3)<3)  
       sample_test = imresize(sample_test,[32 32]);
       else
       sample_test = imresize(rgb2gray(sample_test),[32 32]);
       end
       
       labels_test = [labels_test;0]; 
       %h = hist(double(sample_test(:)'),256);h = h/length(sample_test(:));%Histogram
       features_test = [features_test;sample_test(:)'];
end
%% Loading test images of cat and constructing training feature vector and training labels
for j = 1:length(files4)
    
       image_path = fullfile(files4(i).folder, files4(i).name);
       sample_test = imread(image_path);        
       % RGB to Gray
       if(size(sample_test,3)<3)  
       sample_test = imresize(sample_test,[32 32]);
       else
       sample_test = imresize(rgb2gray(sample_test),[32 32]);
       end
       
       labels_test = [labels_test;1];       
       %h = hist(double(sample_test(:)'),256);h = h/length(sample_test(:));%Histogram
       features_test = [features_test;sample_test(:)'];
end
    
%% KNN Classifier
acc=[];
for k=1:1:100
%k=5;
predict_label=0;

for i=1:size(features_test,1)
    
    predict_label(i) = knn_classifier(features_train, labels_train, features_test(i,:),k);
end

accuracy = 100*sum(predict_label' == labels_test)/length(labels_test);
acc=[acc accuracy];
end
