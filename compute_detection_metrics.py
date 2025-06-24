# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:30:00 2025
@author: mukherjees9

1) Compute ROC‐AUC & optimal Youden’s J threshold for:
     - AI-predicted area
     - Each clinician’s annotated area
2) Binarize each method’s predictions using its threshold
3) Compute per-image Dice (DSC=1.0 when both GT & prediction empty)
4) Report AUCs, thresholds, and mean±SD Dice for all methods
5) Save results and plot ROC curves—all outputs in saveLoc
"""
from scipy.stats import norm

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from sklearn.metrics import roc_auc_score, roc_curve

def file_exists_safe(path):
    p = os.path.normpath(path.strip())
    if len(p) >= 260:
        p = r'\\?\{}'.format(p)
    return p

# — Dice coefficient helper —
def dice_coef(m1, m2):
    s = (m1>0).sum() + (m2>0).sum()
    return 1.0 if s == 0 else 2.0 * np.sum((m1>0)&(m2>0)) / s

# — Paths & constants —
gt_folder     = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings\dylan'
pred_folder   = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\Segmentation'
gtBaseLoc     = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings'
saveLoc       = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\SegmentationOverlapWithGT\clinicians_gradings\guymer_based_dice'
PIX_SPACING   = 0.01174  # mm per pixel

os.makedirs(saveLoc, exist_ok=True)

# — Clinician folders —
clinicians        = ['alisa','leon','marco','mehdi']
# Map each original name to a grader label
label_map = {
    "Alisa":  "Grader 1",
    "Leon":   "Grader 2",
    "Marco":  "Grader 3",
    "Mehdi":  "Grader 4"
}

clinician_folders = {c: os.path.join(gtBaseLoc, c) for c in clinicians}

# — Collect data —
records = []
gt_masks   = {}
ai_masks   = {}
clin_masks = {c: {} for c in clinicians}

for fname in os.listdir(gt_folder):
    if not fname.lower().endswith('.tif'):
        continue

    gt_path = os.path.join(gt_folder, fname)
    pr_path = os.path.join(pred_folder, fname)
    if not os.path.exists(pr_path):
        continue

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
    if gt is None or pr is None:
        continue

    gt_masks[fname] = gt
    ai_masks[fname] = pr

    gt_label = int((gt > 0).sum() > 0)
    ai_area  = (pr > 0).sum() * PIX_SPACING**2

    row = {'image': fname, 'gt_label': gt_label, 'AI_area_mm2': ai_area}

    for c in clinicians:
        found = None
        for fn in os.listdir(clinician_folders[c]):
            if fn.lower().endswith('.png') and fn.rsplit('__',1)[-1][:-4]+'.tif' == fname:
                found = fn
                break
        if found:
            cm = cv2.imread(file_exists_safe(os.path.join(clinician_folders[c], found)),
                            cv2.IMREAD_GRAYSCALE)
            clin_masks[c][fname] = cm
            row[f'{c}_area_mm2'] = (cm > 0).sum() * PIX_SPACING**2
        else:
            row[f'{c}_area_mm2'] = np.nan

    records.append(row)

df = pd.DataFrame(records)

# — Compute AUC, ROC curves & Youden’s J thresholds for all methods —
methods = [('AI_area_mm2','AI Model')] + [(f'{c}_area_mm2', c.capitalize()) for c in clinicians]
auc_results      = {}
thresholds       = {}

plt.figure(figsize=(6,6))
for col, label in methods:
    scores = df[col].fillna(0)
    y_true = df['gt_label']

    # ROC & AUC
    fpr, tpr, thr = roc_curve(y_true, scores)
    auc_ = roc_auc_score(y_true, scores)
    auc_results[label] = auc_

    # Youden's J
    J = tpr - fpr
    ix = np.argmax(J)
    opt_thr = thr[ix]
    thresholds[label] = opt_thr

    # — compute Wald-test p-value for AUC vs. 0.5 —
    n1 = int(y_true.sum())             # # positives
    n2 = int(len(y_true) - n1)         # # negatives
    Q1 = auc_ / (2 - auc_)
    Q2 = 2 * auc_**2 / (1 + auc_)
    se_auc = np.sqrt((auc_*(1-auc_) +
                      (n1-1)*(Q1-auc_**2) +
                      (n2-1)*(Q2-auc_**2)) /
                     (n1 * n2))
    z = (auc_ - 0.5) / se_auc
    pval = 2 * (1 - norm.cdf(abs(z)))

    # Use the grader label instead of the raw name
    grader_label = label_map.get(label, label)
    
    # Plot
    # plt.plot(fpr, tpr, '--', label=f"{label} (AUC={auc_:.2f}, Thr={opt_thr:.2f})")
    plt.plot(fpr, tpr, '--', label=f"{grader_label} (AUC={auc_:.2f}, p={pval:.3f})")

# Random baseline
plt.plot([0,1], [0,1], 'k--', label='Random (AUC=0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: RPD Detection by Graders or AI')
plt.legend(loc='lower right', fontsize='small')
plt.tight_layout()

roc_path = os.path.join(saveLoc, 'roc_curves_all_methods.tif')
plt.savefig(roc_path, dpi=500)
plt.show()

# — Compute Dice for each method using its threshold —
dice_results = {}
for col, label in methods:
    thr = thresholds[label]
    dscs = []
    for fname in df['image']:
        gt = gt_masks[fname]
        if label == 'AI Model':
            pr = ai_masks[fname]
        else:
            pr = clin_masks[label.lower()].get(fname)
        
        pr_bin = np.zeros_like(pr) if (pr>0).sum()*PIX_SPACING**2 < thr else pr
        dscs.append(dice_coef(gt, pr_bin))
    dice_results[label] = (np.mean(dscs), np.std(dscs))

# — Print summary —
print("=== AUC & Thresholds ===")
for label in auc_results:
    print(f"{label:12s}: AUC={auc_results[label]:.3f}, Thr={thresholds[label]:.4f} mm²")
print("\n=== Dice (mean ± SD) ===")
for label, (m, sd) in dice_results.items():
    print(f"{label:12s}: {m:.3f} ± {sd:.3f}")

# — Save detailed results —
df.to_excel(os.path.join(saveLoc, 'detection_metrics_all_methods.xlsx'), index=False)
print(f"\n✔ Metrics saved to: {os.path.join(saveLoc, 'detection_metrics_all_methods.xlsx')}")
print(f"✔ ROC plot saved to: {roc_path}")
