#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute ROC‐AUC, standard error, and Wald‐type p‐value vs. chance (AUC0 = 0.5)
for AI model, each grader, and combined graders.
"""

import os
import cv2
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# — Helper for long paths —
def file_exists_safe(path):
    p = os.path.normpath(path.strip())
    return p if len(p) < 260 else r'\\?\{}'.format(p)

# — User paths and settings —
PIX_SPACING    = 0.01174  # mm per pixel
gt_folder      = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\dylan"
pred_folder    = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\Segmentation"
gtBaseLoc      = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings"
clinicians     = ['alisa','leon','marco','mehdi']
label_map      = {"alisa":"Grader 1","leon":"Grader 2","marco":"Grader 3","mehdi":"Grader 4"}
clinician_folders = {c: os.path.join(gtBaseLoc, c) for c in clinicians}

# — 1) Load masks and compute binary labels & continuous scores —
records = []
for fname in os.listdir(gt_folder):
    if not fname.lower().endswith('.tif'):
        continue
    gt_path = os.path.join(gt_folder, fname)
    pr_path = os.path.join(pred_folder, fname)
    if not os.path.exists(pr_path):
        continue
    gt = cv2.imread(file_exists_safe(gt_path), cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(file_exists_safe(pr_path), cv2.IMREAD_GRAYSCALE)
    if gt is None or pr is None:
        continue

    gt_label = int((gt > 0).sum() > 0)
    ai_area = (pr > 0).sum() * PIX_SPACING**2

    row = {'image': fname, 'gt_label': gt_label, 'AI_area_mm2': ai_area}
    # each grader's area
    for c in clinicians:
        area = 0.0
        folder = clinician_folders[c]
        for fn in os.listdir(folder):
            if fn.lower().endswith('.png') and fn.split('__')[-1][:-4]+'.tif' == fname:
                cm = cv2.imread(file_exists_safe(os.path.join(folder, fn)), cv2.IMREAD_GRAYSCALE)
                if cm is not None:
                    area = (cm > 0).sum() * PIX_SPACING**2
                break
        row[f'{c}_area_mm2'] = area

    records.append(row)

import pandas as pd
df = pd.DataFrame(records)
y_true = df['gt_label'].values
n1 = int(y_true.sum())
n2 = len(y_true) - n1

# — 2) AUC and standard error via Hanley & McNeil —
def compute_se_auc(auc, n_pos, n_neg):
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc**2 / (1 + auc)
    return np.sqrt((auc*(1-auc) + (n_pos-1)*(Q1-auc**2) + (n_neg-1)*(Q2-auc**2)) / (n_pos*n_neg))

# — 3) Compute results for AI, graders, and combined —
results = {}

# AI
auc_ai = roc_auc_score(y_true, df['AI_area_mm2'])
se_ai  = compute_se_auc(auc_ai, n1, n2)
# p-value vs chance
z0_ai = (auc_ai - 0.5) / se_ai
p0_ai = 2*(1 - norm.cdf(abs(z0_ai)))
results['AI Model'] = {'auc': auc_ai, 'se': se_ai, 'p_vs_chance': p0_ai}

# Individual graders
for c in clinicians:
    lbl = label_map[c]
    auc_g = roc_auc_score(y_true, df[f'{c}_area_mm2'])
    se_g  = compute_se_auc(auc_g, n1, n2)
    z0_g = (auc_g - 0.5) / se_g
    p0_g = 2*(1 - norm.cdf(abs(z0_g)))
    results[lbl] = {'auc': auc_g, 'se': se_g, 'p_vs_chance': p0_g}

# Combined graders (majority vote ≥2)
vote = (df[[f'{c}_area_mm2' for c in clinicians]] > 0).astype(int).sum(axis=1) >= 2
auc_comb = roc_auc_score(y_true, vote.astype(int))
se_comb  = compute_se_auc(auc_comb, n1, n2)
z0_c = (auc_comb - 0.5) / se_comb
p0_c = 2*(1 - norm.cdf(abs(z0_c)))
results['Combined Graders'] = {'auc': auc_comb, 'se': se_comb, 'p_vs_chance': p0_c}

# — 4) Print results —
print("=== AUC (±SE) and p-value vs. chance ===")
for name, v in results.items():
    print(f"{name:20s}: {v['auc']:.3f} ± {v['se']:.3f} (p={v['p_vs_chance']:.3f})")

