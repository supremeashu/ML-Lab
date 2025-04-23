%clc
clear all

folder_path1 = 'C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\train\dogs';
folder_path2 = 'C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\train\cats';
folder_path5 = 'C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\train\butterfly';

files1 = dir(fullfile(folder_path1, '**', '*.jpg'));
files2 = dir(fullfile(folder_path2, '**', '*.jpg'));
files5 = dir(fullfile(folder_path5, '**', '*.jpg'));

bin=128;
row=128; column=128;
features_train = [];
labels_train = [];
%% Loading training images of dog and constructing training feature vector and training labels
for i = 1:1:length(files1)
    
       image_path = fullfile(files1(i).folder, files1(i).name);
       sample_train = imread(image_path);        
       % RGB to Gray
       if(size(sample_train,3)<3)  
       sample_train = imresize(sample_train,[row column]);
       else
       sample_train = imresize(rgb2gray(sample_train),[row column]);
       end
       
       labels_train = [labels_train;0]; 
       %Normalized Histogram
       h = hist(double(sample_train(:)'),bin);h = h/length(sample_train(:));%Histogram
       features_train = [features_train;h];
end
%% Loading training images of cat and constructing training feature vector and training labels
for j = 1:length(files2)
       image_path = fullfile(files2(j).folder, files2(j).name);
       sample_train = rgb2gray(imread(image_path)); 
       % RGB to Gray 
       if(size(sample_train,3)<3)  
       sample_train = imresize(sample_train,[row column]);
       else
       sample_train = imresize(rgb2gray(sample_train),[row column]);
       end
       
       labels_train = [labels_train;1];   
       h = hist(double(sample_train(:)'),bin);h=h/length(sample_train(:));%%Histogram
       features_train = [features_train;h];
end

%% Loading training images of butterfly and constructing training feature vector and training labels
for j = 1:length(files5)
       image_path = fullfile(files5(j).folder, files5(j).name);
       sample_train = rgb2gray(imread(image_path)); 
       % RGB to Gray 
       if(size(sample_train,3)<3)  
       sample_train = imresize(sample_train,[row column]);
       else
       sample_train = imresize(rgb2gray(sample_train),[row column]);
       end
       
       labels_train = [labels_train;2];   
       h = hist(double(sample_train(:)'),bin);h=h/length(sample_train(:));%%Histogram
       features_train = [features_train;h];
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
folder_path3 = 'C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\test\dogs';
folder_path4 = 'C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\test\cats';
folder_path6 = 'C:\D folder\Programming\programs\my codes\ML codes\knn classifier\Dog-Cat Dataset\test\butterfly';

files3 = dir(fullfile(folder_path3, '**', '*.jpg'));
files4 = dir(fullfile(folder_path4, '**', '*.jpg'));
files6 = dir(fullfile(folder_path6, '**', '*.jpg'));

total_images = length(files3)+length(files4);

features_test = [];
labels_test = [];
%% Loading test images of dog and constructing training feature vector and training labels
for i = 1:length(files3)
    
       image_path = fullfile(files3(i).folder, files3(i).name);
       sample_test = imread(image_path);        
       % RGB to Gray
       if(size(sample_test,3)<3)  
       sample_test = imresize(sample_test,[row column]);
       else
       sample_test = imresize(rgb2gray(sample_test),[row column]);
       end
       
       labels_test = [labels_test;0]; 
       h = hist(double(sample_test(:)'),bin);h = h/length(sample_test(:));%Histogram
       features_test = [features_test;h];
end
%% Loading test images of cat and constructing training feature vector and training labels
for j = 1:length(files4)
    
       % image_path = fullfile(files4(i).folder, files4(i).name);
       image_path = fullfile(files4(j).folder, files4(j).name);
       sample_test = imread(image_path);        
       % RGB to Gray
       if(size(sample_test,3)<3)  
       sample_test = imresize(sample_test,[row column]);
       else
       sample_test = imresize(rgb2gray(sample_test),[row column]);
       end
       
       labels_test = [labels_test;1];       
       h = hist(double(sample_test(:)'),bin);h = h/length(sample_test(:));%Histogram
       features_test = [features_test;h];
end
%% Loading test images of butterfly and constructing training feature vector and training labels
for j = 1:length(files6)
    
       % image_path = fullfile(files6(i).folder, files6(i).name);
       image_path = fullfile(files6(j).folder, files6(j).name);
       sample_test = imread(image_path);        
       % RGB to Gray
       if(size(sample_test,3)<3)  
       sample_test = imresize(sample_test,[row column]);
       else
       sample_test = imresize(rgb2gray(sample_test),[row column]);
       end
       
       labels_test = [labels_test;2];       
       h = hist(double(sample_test(:)'),bin);h = h/length(sample_test(:));%Histogram
       features_test = [features_test;h];
end
%%
r = randperm(length(labels_test));

%% SVM Classifier
% Standardization
m = mean(features_train); st = std(features_train);
features_train0 = (features_train-m)./(st);
features_test0 = (features_test-m)./(st);

%PCA for dimensionality reduction
[COEFF, SCORE_train, LATENT] = pca(features_train0);
ac=[];
% for s=1:1:50
features_train1 = (features_train0-mean(features_train0))*COEFF(:,1:1);  
features_test1 = (features_test0-mean(features_train0))*COEFF(:,1:1);

%opts = struct('AcquisitionFunctionName', 'expected-improvement-plus'); % Optimization settings

svmTemplate = templateSVM('KernelFunction', 'rbf', 'KernelScale', 2,'BoxConstraint', 0.1,'Standardize', false);

SVMModelOptimized = fitcecoc(features_train1, labels_train,  'Learners', svmTemplate);
                      
predict_label = predict(SVMModelOptimized, features_test1(r(1:100),:));

accuracy = 100*sum(predict_label == labels_test(r(1:100)'))/length(labels_test(r(1:100)'));
ac=[ac accuracy];
% close all
% end
%%
[p,q]= max(ac);
PC_opt = q;
features_train2 = (features_train0-mean(features_train0))*COEFF(:,1:PC_opt);  
features_test2 = (features_test0-mean(features_train0))*COEFF(:,1:PC_opt);

SVMModelOptimized = fitcecoc(features_train2, labels_train, 'Learners', svmTemplate);
predict_label = predict(SVMModelOptimized, features_test2(r(101:end),:));

accuracy = 100*sum(predict_label == labels_test(r(101:end)'))/length(labels_test(r(101:end)'))
C = confusionmat(labels_test(r(101:end)'),predict_label);
confusionchart(C)

 
