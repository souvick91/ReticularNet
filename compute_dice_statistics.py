# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:44:20 2025

@author: mukherjees9
"""

import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from itertools import combinations
from scipy.stats import wilcoxon
import pdb

def load_labels(grader, folder, dylan_files, onlyRPDpos, dylan_mask_data):
    """Load binary label masks from a folder. If not Dylan, filter to match Dylan's files."""

    # Step 1: Build map from core image name to full path
    file_map = {}
    
    if grader == 'Dylan':
        # all Dylan ground-truth TIFFs
        files = sorted(glob(os.path.join(folder, "*.tif")))
        for f in files:
            name = os.path.basename(f)
            file_map[name] = f
        selected_keys = sorted(file_map.keys())

    elif grader == "AI Model":
        # only AI predictions for images Dylan graded
        files = sorted(glob(os.path.join(folder, "*.tif")))
        for f in files:
            name = os.path.basename(f)
            if dylan_files and name in dylan_files:
                file_map[name] = f
        selected_keys = sorted(file_map.keys())
        
    else:
        all_files = glob(os.path.join(folder, "*.png"))
        for f in all_files:
            base = os.path.basename(os.path.normpath(f))
            if "__" in base:
                core_name = base.split("__")[-1]
                file_map[core_name] = f
        if dylan_files:
            selected_keys = sorted([k for k in file_map.keys() if k.replace('.png', '.tif') in dylan_files])
        else:
            selected_keys = sorted(file_map.keys())

    # Step 2: Load masks
    masks = []
    for k in selected_keys:
        f = file_map[k]
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 0).astype(np.uint8)
    
        # Skip if both Dylan mask is empty
        if onlyRPDpos and dylan_mask_data is not None:
            key_match = k.replace('.png', '.tif')
            if key_match in dylan_mask_data["keys"]:
                idx = dylan_mask_data["keys"].index(key_match)
                if np.sum(dylan_mask_data["masks"][idx]) == 0:
                    continue       
        masks.append(binary_mask)
    if grader in ('Dylan', 'AI Model'):
        return masks, selected_keys
    else:
        return masks

def dice_coefficient(mask1, mask2):
    """Compute Dice Similarity Coefficient (DSC) between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    return (2. * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-8)  # Avoid division by zero

# Define paths to each grader's labels
includesMarco = True
onlyRPDpos = True
saveLoc = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\results'

if includesMarco:
    if onlyRPDpos:
        xlsxSaveName = 'dice_statistics_with_Marco_onlyRPDPos.xlsx'
    else:
        xlsxSaveName = 'dice_statistics_with_Marco_all.xlsx'
        
    grader_folders = {
        "Alisa": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\alisa",
        "Leon": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\leon",
        "Marco": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\marco",
        "Mehdi": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\mehdi",
        "Dylan": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\dylan",
        "AI Model":    r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\Segmentation",  
    }
    numMajorVoting = 3
else:
    if onlyRPDpos:
        xlsxSaveName = 'dice_statistics_without_Marco_onlyRPDPos.xlsx'
    else:
        xlsxSaveName = 'dice_statistics_without_Marco_all.xlsx'
        
    grader_folders = {
        "Alisa": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\alisa",
        "Leon": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\leon",
        "Mehdi": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\mehdi",
        "Dylan": r"\\?\Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\dylan",
        "AI Model":    r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\Segmentation",  
    }
    numMajorVoting = 2

gt_folder   = grader_folders["Dylan"]
gt_files    = [os.path.basename(f) for f in glob(os.path.join(gt_folder, "*.tif"))]
dylan_masks, dylan_keys = load_labels(
    "Dylan", gt_folder,
    dylan_files=None, onlyRPDpos=False, dylan_mask_data=None
)

# now load AI masks too
ai_folder    = grader_folders["AI Model"]
ai_masks, _  = load_labels(
    "AI Model", ai_folder,
    dylan_files=gt_files, onlyRPDpos=False, dylan_mask_data=None
)

dylan_mask_data = {
    "masks": dylan_masks,
    "keys": dylan_keys
}

# Load the binary masks
grader_masks = {}
for grader, folder in grader_folders.items():
    if grader in ("Dylan"):
        masks, _ = load_labels(grader, folder, dylan_files=None,
                               onlyRPDpos=onlyRPDpos, dylan_mask_data=dylan_mask_data)
    elif grader in ("AI Model"):
        masks, _ = load_labels(grader, folder, dylan_files=gt_files,
                               onlyRPDpos=onlyRPDpos, dylan_mask_data=dylan_mask_data)        
    else:
        masks = load_labels(grader, folder, gt_files, onlyRPDpos, dylan_mask_data)
    grader_masks[grader] = masks

# Check all graders have the same number of labels
num_samples = len(next(iter(grader_masks.values())))
assert all(len(masks) == num_samples for masks in grader_masks.values()), "Mismatch in number of samples"

# Compute total agreement (majority vote)
total_masks = []
for i in range(num_samples):
    stack = np.stack([grader_masks[grader][i] for grader in grader_folders.keys()])
    total_mask = (np.sum(stack, axis=0) >= numMajorVoting).astype(np.uint8)  # Majority voting
    total_masks.append(total_mask)
    
# Compute total agreement (full agreement)
total_masks_fa = []
for i in range(num_samples):
    stack = np.stack([grader_masks[grader][i] for grader in grader_folders.keys()])
    total_mask = (np.sum(stack, axis=0) == len(grader_folders)).astype(np.uint8)  # Full agreement
    total_masks_fa.append(total_mask)
    

# Compute pairwise Dice similarity
graders = list(grader_folders.keys())
dice_scores = pd.DataFrame(index=graders, columns=graders + ["Total (majority voting)"] + ["Total (full agreement)"], dtype=str)

for grader1, grader2 in combinations(graders, 2):
    per_image_dice = [dice_coefficient(m1, m2) for m1, m2 in zip(grader_masks[grader1], grader_masks[grader2])]
    mean_dice = np.mean(per_image_dice)
    std_dice = np.std(per_image_dice)
    formatted = f"{mean_dice:.2f} ± {std_dice:.2f}"
    dice_scores.loc[grader1, grader2] = formatted
    dice_scores.loc[grader2, grader1] = formatted  # Symmetric matrix

# Compute Dice similarity between each grader and the total agreement (majority vote)
for grader in graders:
    per_image_dice = [dice_coefficient(m1, m2) for m1, m2 in zip(grader_masks[grader], total_masks)]
    mean_dice = np.mean(per_image_dice)
    std_dice = np.std(per_image_dice)
    dice_scores.loc[grader, "Total (majority voting)"] = f"{mean_dice:.3f} ± {std_dice:.3f}"
       
# Compute Dice similarity between each grader and the total agreement (full agreement)
for grader in graders:
    per_image_dice = [dice_coefficient(m1, m2) for m1, m2 in zip(grader_masks[grader], total_masks_fa)]
    mean_dice = np.mean(per_image_dice)
    std_dice = np.std(per_image_dice)
    dice_scores.loc[grader, "Total (full agreement)"] = f"{mean_dice:.3f} ± {std_dice:.3f}" 

# Compute Dice similarity for all 4 graders together
all_grader_masks = [grader_masks[grader] for grader in graders]

# Compute Dice coefficient across all 4 graders per image (majority voting)
all_grader_dice_mv = np.mean([
    dice_coefficient(
        (np.sum([masks[i] for masks in all_grader_masks], axis=0) >=numMajorVoting ).astype(np.uint8),  # Intersection of all 2 graders
        (np.sum([masks[i] for masks in all_grader_masks], axis=0) > 0).astype(np.uint8)  # Union of all 4 graders
    )
    for i in range(num_samples)
])

# Display results
mean_mv = np.mean(all_grader_dice_mv)
std_mv = np.std(all_grader_dice_mv)

if includesMarco:
    dice_scores.loc["All Graders Dice (majority voting)", :] = [None, None, None, None, None, None, f"{mean_mv:.3f} ± {std_mv:.3f}", None]
else:
    dice_scores.loc["All Graders Dice (majority voting)", :] = [None, None, None, None, None, f"{mean_mv:.3f} ± {std_mv:.3f}", None]

# Compute Dice coefficient across all 4 graders per image
all_grader_dice_fa = np.mean([
    dice_coefficient(
        (np.sum([masks[i] for masks in all_grader_masks], axis=0) == len(all_grader_masks)).astype(np.uint8),  # Intersection of all 4 graders
        (np.sum([masks[i] for masks in all_grader_masks], axis=0) > 0).astype(np.uint8)  # Union of all 4 graders
    )
    for i in range(num_samples)
])

mean_fa = np.mean(all_grader_dice_fa)
std_fa = np.std(all_grader_dice_fa)

# Display results
if includesMarco:
    dice_scores.loc["All Graders Dice (full agreement)", :] = [None, None, None, None, None, None, None, f"{mean_fa:.3f} ± {std_fa:.3f}"]
else:
    dice_scores.loc["All Graders Dice (full agreement)", :] = [None, None, None, None, None, None, f"{mean_fa:.3f} ± {std_fa:.3f}"]

# dice_scores = dice_scores*1.0

print(dice_scores)

# Rename Dylan → Ground Truth in both index and columns
dice_scores.rename(
    index={'Dylan': 'Ground Truth'},
    columns={'Dylan': 'Ground Truth'},
    inplace=True
)

# Save DataFrame to an Excel file
dice_scores.to_excel(os.path.join(saveLoc, xlsxSaveName), index=True)

human = ["Alisa", "Leon", "Marco", "Mehdi", "Dylan", "AI Model"]
records = []

for g1, g2 in combinations(human, 2):
    # DSC between the two graders
    dsc_pair = np.array([
        dice_coefficient(grader_masks[g1][i], grader_masks[g2][i])
        for i in range(num_samples)
    ])
    # DSC of g1 vs Ground Truth
    dsc_g1_gt = np.array([
        dice_coefficient(grader_masks[g1][i], dylan_masks[i])
        for i in range(num_samples)
    ])
    # DSC of g1 vs AI
    dsc_g1_ai = np.array([
        dice_coefficient(grader_masks[g1][i], ai_masks[i])
        for i in range(num_samples)
    ])
    
    # compute means & stds
    mean_pair, std_pair = dsc_pair.mean(),    dsc_pair.std()
    mean_gt,   std_gt   = dsc_g1_gt.mean(),   dsc_g1_gt.std()
    mean_ai,   std_ai   = dsc_g1_ai.mean(),   dsc_g1_ai.std()    

    W_gt, p_gt = wilcoxon(dsc_pair, dsc_g1_gt)
    W_ai, p_ai = wilcoxon(dsc_pair, dsc_g1_ai)

    records.append({
        'Grader Pair':      f"{g1}–{g2}",
        'DSC pair (mean±SD)':    f"{mean_pair:.3f}±{std_pair:.3f}",     
        'W vs GT (g1)':     round(W_gt,2),
        'p vs GT (g1)':     round(p_gt, 3),
        'W vs AI (g1)':     round(W_ai,2),
        'p vs AI (g1)':     round(p_ai, 3),
    })

# ─── compute per‐grader aggregate inter‐grader DSC ────────────────────────────
grader_list = ["Alisa","Leon","Marco","Mehdi"]
for g in grader_list:
    # collect per‐image mean DSC of g vs all *other* graders
    others = [h for h in grader_list if h != g]
    gg = np.array([
        np.mean([dice_coefficient(grader_masks[g][i], grader_masks[h][i])
                 for h in others])
        for i in range(num_samples)
    ])
    mean_gg, std_gg = gg.mean(), gg.std()
    # now test gg vs GT and vs AI
    dsc_g_gt = np.array([dice_coefficient(grader_masks[g][i], dylan_masks[i])
                         for i in range(num_samples)])
    dsc_g_ai = np.array([dice_coefficient(grader_masks[g][i], ai_masks[i])
                         for i in range(num_samples)])
    W_gt_agg, p_gt_agg = wilcoxon(gg, dsc_g_gt)
    W_ai_agg, p_ai_agg = wilcoxon(gg, dsc_g_ai)
    
    # append as a special “aggregate” row
    records.append({
        'Grader Pair':         f"{g} (aggregate)",
        'DSC pair (mean±SD)':  f"{mean_gg:.3f}±{std_gg:.3f}",
        'W vs GT (agg)':      round(W_gt_agg,2),
        'p vs GT (agg)':      round(p_gt_agg,3),
        'W vs AI (agg)':      round(W_ai_agg,2),
        'p vs AI (agg)':      round(p_ai_agg,3),
         'W vs GT (g1)':        "",
         'p vs GT (g1)':        "",
         'W vs AI (g1)':        "",
         'p vs AI (g1)':        "",
    })
    
stat_df = pd.DataFrame(records)
stat_df.to_excel(os.path.join(saveLoc, 'grader_pair_wilcoxon_vs_GT_AI_v02.xlsx'),
                 index=False)
print("Saved pairwise Wilcoxon results → grader_pair_wilcoxon_vs_GT_AI_v02.xlsx")