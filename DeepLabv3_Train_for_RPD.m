
% clear all; close all; clc;

%%
% UPDATE THE FAF IMAGE LOCATION AND GT CONTOUR LOCATION BELOW
ImageLocation = 'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\TrainImages';
GTLocation = 'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\TrainMasks';
pretrainedModel = 'Z:\Souvick\Projects\RPDSegmentation\IRSegmentationCode\deeplabv3plusResnet18CamVid_v2.mat';
%%

imds = imageDatastore(ImageLocation);
classNames = ["ba", "ga"];
pixelLabelIDs = [0 255];
pxds = pixelLabelDatastore(GTLocation,classNames,pixelLabelIDs);

imageSize = [768 768 3];
numClasses = numel(classNames);
lgraph = deeplabv3plusLayers(imageSize,numClasses,'resnet18');
layers = lgraph.Layers;
connections = lgraph.Connections;

% layers(100).Classes = classNames;
% layers(100).ClassWeights = [1, 3];

diceWeight = 0.5;
classWeights = [1, 5.0];  % [ba, ga]

lossLayer = combinedCrossEntropyDiceLoss("classification", diceWeight, classWeights);
layers(100) = lossLayer;

% layers(72).DilationFactor = [6 6];
% layers(75).DilationFactor = [12 12];
% layers(78).DilationFactor = [18 18];

lgraph = createLgraphUsingConnections(layers,connections);

cds = combine(imds,pxds);
tds = transform(cds, @(data)preprocessTrainingData(data));
opts = trainingOptions('sgdm',...
    'InitialLearnRate', 0.0001,...
    'LearnRateSchedule', 'piecewise',...
    'LearnRateDropFactor', 1,...
    'LearnRateDropPeriod', 25,...
    'MiniBatchSize',2,...
    'MaxEpochs', 10);

net_finetuned_for_rpd = trainNetwork(tds, lgraph,opts);
save net_finetuned_for_rpd net_finetuned_for_rpd

function data = preprocessTrainingData(data)
% Resize the training image and associated pixel label image.
    for i = 1:size(data,1)
        tform = randomAffine2d(...
            XReflection=true,...
            YReflection=true,...      
            Rotation=[-5 5], ...
            XTranslation=[-5 5], ...
            YTranslation=[-5 5], ...
            Scale=[0.90 1.10], ...
            XShear=[0 5], ...
            YShear=[0 5] ...
            );
            
        
        % Center the view at the center of image in the output space while
        % allowing translation to move the output image out of view.
        rout = affineOutputView(size(data{i,1}), tform, BoundsStyle='centerOutput');
        
        % Warp the image and pixel labels using the same transform.
        data{i,1} = imwarp(data{i,1}, tform, OutputView=rout);
        data{i,2} = imwarp(data{i,2}, tform, OutputView=rout);

        % If grayscale, convert to RGB for color jitter
        if size(data{i,1}, 3) == 1
            data{i,1} = repmat(data{i,1}, [1 1 3]);  % replicate single channel to RGB
        end

        % Apply brightness and contrast jitter (only for RGB images)
        data{i,1} = jitterColorHSV(data{i,1}, ...
            'Brightness', 0.15, ...
            'Contrast', 0.15, ...
            'Saturation', 0.1, ...
            'Hue', 0.05);
    end
end

