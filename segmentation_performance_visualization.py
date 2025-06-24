# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:55:17 2024
Modified on Apr 15 2025

@author: mukherjees9 / revised by ChatGPT
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import cv2
import numpy as np
import copy
import pandas as pd
import pdb

def file_exists_safe(path):
    # Normalize and check UNC prefix if needed
    normalized = os.path.normpath(path.strip())
    if len(normalized) >= 260:
        normalized = r'\\?\{}'.format(normalized)
    return normalized

# Define paths
predLoc = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\Segmentation'
gtBaseLoc = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings'
rawIRLoc = r'Z:\Souvick\Projects\Segment_all_years\IRImageAllFiles'
saveLocBase = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\SegmentationOverlapWithGT\clinicians_gradings'

# Clinical graders' subfolders
graders = ['alisa', 'leon', 'marco', 'mehdi', 'dylan']
allFiles = os.listdir(predLoc)
xyspacing = 0.01174

# Font settings for overlay text
font = cv2.FONT_HERSHEY_SIMPLEX
org = (1400, 20)
fontScale = 0.7
color = (0, 255, 255)
thickness = 2

overall_dsc = []
per_image_dsc = {}
area_diff_by_grader = {}

# Process each grader
for grader in graders:
    print(f"\nProcessing grader: {grader}")
    gt_map = {}

    grader_folder = os.path.join(gtBaseLoc, grader)
    saveLoc = os.path.join(saveLocBase, grader)
    os.makedirs(saveLoc, exist_ok=True)

    # Build GT file mapping
    for fname in os.listdir(grader_folder):
        if grader == 'dylan':
            gt_map[fname] = os.path.join(grader_folder, fname)
        else:
            if "__" in fname:
                key_part = fname.rsplit("__", 1)[-1]
                key = key_part.replace('.png', '.tif')  # or '.tif' if that's the format used in predLoc
                gt_map[key] = os.path.join(grader_folder, fname)

    dscAll = []
    pred_area_list = []
    gt_area_list = []
    image_dsc_map = {}

    # if grader=='marco':
    #     pdb.set_trace()
        
    for file in allFiles:
        if file.endswith(".db"):
            continue
        if not os.path.exists(os.path.join(gtBaseLoc, 'dylan', file)):
            print('Skipped   ', file)
            continue

        pred_path = os.path.join(predLoc, file)
        if not os.path.exists(pred_path):
            continue

        predImg = cv2.imread(pred_path, 0) / 255.0

        gt_path = gt_map.get(file)
        
        gt_path = file_exists_safe(gt_path)
        
        if not gt_path or not os.path.exists(gt_path):
            continue

        gtImg = cv2.imread(gt_path, 0)
        gtImg[gtImg > 0] = 1

        ir_path = os.path.join(rawIRLoc, file.replace('.tif', '.png'))
        if not os.path.exists(ir_path):
            continue

        IRImg = cv2.imread(ir_path)
        
        # pdb.set_trace()

        # Compute DSC
        dscImg = 2 * (np.sum(predImg * gtImg)) / (np.sum(predImg + gtImg)) if np.sum(predImg + gtImg) > 0 else 0
        
        ### Dylan mask
        gtDylanMask = cv2.imread(os.path.join(gtBaseLoc, 'dylan', file), 0)
        gtDylanMask[gtDylanMask>0] = 1
        
        if np.sum(gtDylanMask) > 0:
            dscAll.append(dscImg)
            image_dsc_map[file] = dscImg

        # Area computation
        pred_area = np.sum(predImg) * xyspacing ** 2
        gt_area = np.sum(gtImg) * xyspacing ** 2
        pred_area_list.append(pred_area)
        gt_area_list.append(gt_area)

        # Overlay visualization
        predImgStack = copy.deepcopy(IRImg)
        intersectedPixs = predImg * gtImg

        predImgStack[gtImg == 1] = [255, 255, 0]         # Cyan for GT
        predImgStack[predImg == 1] = [255, 0, 255]       # Magenta for prediction
        predImgStack[intersectedPixs == 1] = [0, 255, 255]  # Yellow for overlap

        predImgStack = np.concatenate([IRImg, predImgStack], axis=1)
        predImgStack = cv2.putText(predImgStack, 'DSC ' + str(round(dscImg, 2)), org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imwrite(os.path.join(saveLoc, file), predImgStack)

    # Store per-image DSC
    per_image_dsc[grader] = image_dsc_map

    # Summary
    dscAll = np.array(dscAll)
    print(f"{grader}: Dice similarity coefficient  {np.mean(dscAll):.4f} ± {np.std(dscAll):.4f}")
    overall_dsc.append((grader, np.mean(dscAll), np.std(dscAll)))

    # Scatter + regression plot
    if gt_area_list and pred_area_list:
        pred_array = np.array(pred_area_list)
        gt_array = np.array(gt_area_list)

        plt.figure(figsize=(6, 6))
        plt.scatter(gt_array, pred_array, alpha=0.7)
        slope, intercept, r_value, p_value, _ = stats.linregress(gt_array, pred_array)
        plt.plot(gt_array, slope * gt_array + intercept, 'r', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
        plt.xlabel('GT Area (mm$^2$)')
        plt.ylabel('Predicted Area (mm$^2$)')
        plt.title(f'{grader}: Pred vs GT Area\nR={r_value:.2f}, p={p_value:.4f}')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(saveLoc, f'{grader}_xy_area_plot.tif'), dpi=2000)
        plt.close()

        # Bland-Altman
        avg = (pred_array + gt_array) / 2
        diff = pred_array - gt_array
        area_diff_by_grader[grader] = diff.tolist()

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        plt.figure(figsize=(6, 4))
        plt.scatter(avg, diff, alpha=0.6)
        plt.axhline(mean_diff, color='gray', linestyle='--')
        plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--')
        plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--')
        plt.xlabel('Average of GT and Prediction Area (mm$^2$)')
        plt.ylabel('Prediction - GT Area (mm$^2$)')
        plt.title(f'{grader}: Bland-Altman Plot (mm$^2$)')
        plt.tight_layout()
        plt.savefig(os.path.join(saveLoc, f'{grader}_bland_altman.tif'), dpi=2000)
        plt.close()

        # Violin plot
        plt.figure(figsize=(4, 6))
        sns.violinplot(data=diff, orient='v', inner='point')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f'{grader}: Violin Plot of Area Difference\n(Pred - GT) (mm$^2$)')
        plt.ylabel('Difference in Area (mm$^2$)')
        plt.tight_layout()
        plt.savefig(os.path.join(saveLoc, f'{grader}_violin_area_diff.tif'), dpi=2000)
        plt.close()

# Mapping grader names to display names
grader_display_names = {
    'alisa': 'Grader 1',
    'leon': 'Grader 2',
    'marco': 'Grader 3',
    'mehdi': 'Grader 4',
    'dylan': 'Ground Truth'
}

# Combine all differences into one violin plot with new labels
all_diffs, all_labels = [], []
for grader, diffs in area_diff_by_grader.items():
    display_name = grader_display_names.get(grader, grader)  # fallback to original if not mapped
    all_diffs.extend(diffs)
    all_labels.extend([display_name] * len(diffs))

plt.figure(figsize=(10, 6))
sns.violinplot(x=all_labels, y=all_diffs, inner="point",
               order=['Grader 1', 'Grader 2', 'Grader 3', 'Grader 4', 'Ground Truth'])
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('Predicted – Annotated Area (mm$^2$)')
plt.title('Violin Plot of Area Differences\n(Predicted vs Annotated) Across All Graders')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(saveLocBase, 'combined_violin_area_diff.png'), dpi=600)
plt.close()


# Save per-image DSCs to CSV
df_out = []
for grader, image_scores in per_image_dsc.items():
    for fname, dsc in image_scores.items():
        df_out.append({'image': fname, 'grader': grader, 'DSC': dsc})

df_out = pd.DataFrame(df_out)
df_out.to_csv(os.path.join(saveLocBase, 'per_image_dsc_by_grader.csv'), index=False)
