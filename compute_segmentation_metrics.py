# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:15:25 2025

@author: mukherjees9
"""

import os
import cv2
import numpy as np
import pandas as pd
import copy
import pdb

def file_exists_safe(path):
    normalized = os.path.normpath(path.strip())
    if len(normalized) >= 260:
        normalized = r'\\?\{}'.format(normalized)
    return normalized

def compute_dice(mask1, mask2):
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)
    inter = np.sum(m1 * m2)
    total = np.sum(m1) + np.sum(m2)
    return 2 * inter / total if total > 0 else np.nan

def compute_lesion_metrics(grader, mask, pix_spacing, fname, overlay_dir):
    """
    Returns:
      lesion_count,
      total_area_mm2,
      contour_area_mm2
    """
    bin_mask = (mask > 0).astype(np.uint8)
    # 1) Lesion count
    n_labels, _ = cv2.connectedComponents(bin_mask)
    lesion_count = n_labels - 1

    # 2) Total lesion area (pixels → mm²)
    total_area_pixels = np.sum(bin_mask)
    total_area_mm2 = total_area_pixels * (pix_spacing ** 2)

    # 3) Contour area via convex hull of all external contours
    pred_bin = bin_mask * 255
    pred_bin_org = copy.deepcopy(pred_bin)
    
    # --- START minimal additions ---
    # 0a) Dilate stray points so they merge into the main cluster
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    pred_bin = cv2.dilate(pred_bin, dil_kernel, iterations=1)

    # 0b) Then close larger gaps (as before)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_mask = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, close_kernel)
    # --- END minimal additions ---

    # find the outer contour on the closed & dilated mask
    # (adjust unpacking for your OpenCV version)
    cnts = cv2.findContours(closed_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts)==2 else cnts[1]

    if contours:
        # pick the largest contour
        # main_contour = max(contours, key=cv2.contourArea)

        # draw it on the ORIGINAL pred_bin so you see the real mask boundary + hull
        overlay = cv2.cvtColor(pred_bin_org, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay,  contours, -1, (0,0,255), 2)

        out_name = f"{grader}_{os.path.splitext(fname)[0]}_hull.tif"
        cv2.imwrite(os.path.join(overlay_dir, out_name), overlay)

        # sum the area of *each* external contour
        total_contour_px = sum(cv2.contourArea(c) for c in contours)
        contour_area_mm2 = total_contour_px * (pix_spacing**2)

    else:
        contour_area_mm2 = np.nan

    return lesion_count, total_area_mm2, contour_area_mm2

# === Paths & settings ===
predLoc = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\Segmentation'
gtBaseLoc = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings'
dylan_folder = os.path.join(gtBaseLoc, 'dylan')
graders = ['alisa', 'leon', 'marco', 'mehdi']
overlay_dir = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\SegmentationOverlapWithGT\clinicians_gradings\hull_overlays'
os.makedirs(overlay_dir, exist_ok=True)
save_excel_path = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\SegmentationOverlapWithGT\clinicians_gradings\metrics_vs_groundTruth.xlsx'

# pixel spacing in mm
PIX_SPACING = 0.01174

# build filename maps for each human grader
grader_maps = {}
for grader in graders:
    folder = os.path.join(gtBaseLoc, grader)
    files = os.listdir(folder)
    m = {}
    for f in files:
        if '__' in f:
            key = f.rsplit('__', 1)[-1].replace('.png', '.tif')
            m[key] = os.path.join(folder, f)
    grader_maps[grader] = m

# list of all GT files
all_files = [f for f in os.listdir(dylan_folder) if f.endswith('.tif')]

results = []

for grader in graders + ['ai_model']:
    for fname in all_files:
        gt_path = os.path.join(dylan_folder, fname)
        if not os.path.exists(gt_path):
            continue

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None or gt_mask.sum() == 0:
            continue

        # decide test_path
        if grader == 'ai_model':
            test_path = os.path.join(predLoc, fname)
        else:
            test_path = grader_maps[grader].get(fname)

        test_path = file_exists_safe(test_path)
        if not test_path or not os.path.exists(test_path):
            continue

        test_mask = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        if test_mask is None:
            print(f"Error reading {fname} for grader {grader}")
            continue

        # compute metrics
        dsc = compute_dice(gt_mask, test_mask)

        gt_count, gt_area, gt_contour = compute_lesion_metrics(grader, gt_mask, PIX_SPACING, fname, overlay_dir)
        test_count, test_area, test_contour = compute_lesion_metrics(grader, test_mask, PIX_SPACING, fname, overlay_dir)

        results.append({
            'image': fname,
            'grader': 'AI Model' if grader == 'ai_model' else grader.capitalize(),
            'dice_vs_ground_truth': round(dsc, 4),
            'gt_lesion_count': gt_count,
            'grader_lesion_count': test_count,
            'gt_total_area_mm2': round(gt_area, 4),
            'grader_total_area_mm2': round(test_area, 4),
            'gt_contour_area_mm2': round(gt_contour, 4),
            'grader_contour_area_mm2': round(test_contour, 4),
        })

# Save all metrics
df = pd.DataFrame(results)
# … after you've built your DataFrame `df` …
df.to_excel(save_excel_path, index=False)
print(f"\n✅ Metrics saved to:\n{save_excel_path}")
