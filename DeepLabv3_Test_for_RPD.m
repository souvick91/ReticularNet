% clear all; close all; clc;
load net_finetuned_for_rpd.mat;

files = dir(['Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\TestImages\*.tif']);
Output = 'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\QualityCheck';
Output2 = 'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\Segmentation';
Output3 = 'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\matrices';

for i =1:length(files)
    I = imread(fullfile(files(i).folder,files(i).name));
    if numel(size(I)) == 2
        I = cat(3, I, I, I);
    end
    I = imresize(I,[768 768]);
    disp(files(i).name)
    [C,score,allScores] = semanticseg(I,net_finetuned_for_rpd);
    % C = semanticseg(I,net);
    X = [0 255];
    N = X(C);
    B = labeloverlay(I,C);
    fileSaveName = fullfile(Output3,append(files(i).name(1:length(files(i).name)-4),'.mat'));
    save(fileSaveName, 'C', 'score', 'allScores')
    imwrite([I B], fullfile(Output, files(i).name))
    imwrite(N, fullfile(Output2, files(i).name))
end



