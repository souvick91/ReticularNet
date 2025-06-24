# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:53:51 2024

@author: mukherjees9
"""

import os
import tifffile as tiff
import cv2
import pdb


maskLoc = r'Z:\Souvick\Projects\RPDSegmentation\Weka_Masked_IR_Segmentations\results\Masks'
saveLoc = r'Z:\Souvick\Projects\RPDSegmentation\Weka_Masked_IR_Segmentations\results\Masks_uint8'

allFiles = os.listdir(maskLoc)

for file in allFiles:
    labelImg = tiff.imread(os.path.join(maskLoc, file))
    labelImg = labelImg.astype('uint8')
    labelImg[labelImg==1] = 255
    
    cv2.imwrite(os.path.join(saveLoc, file), labelImg)
