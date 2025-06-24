# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:58:50 2024

@author: mukherjees9
"""

import os
import tifffile as tiff
import cv2
import numpy as np
import pdb


maskLoc = r'Z:\Souvick\Projects\RPDSegmentation\Weka_Masked_IR_Segmentations\results\Images'
saveLoc = r'Z:\Souvick\Projects\RPDSegmentation\Weka_Masked_IR_Segmentations\results\ImagesRGB'

allFiles = os.listdir(maskLoc)

for file in allFiles:
    labelImg = cv2.imread(os.path.join(maskLoc, file))   
    cv2.imwrite(os.path.join(saveLoc, file), labelImg)