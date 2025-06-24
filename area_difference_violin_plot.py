# -*- coding: utf-8 -*-
"""
Standalone script to generate a violin plot of area differences:
  • Human graders’ annotations vs. Dylan (“ground truth”)
  • AI predictions vs. Dylan
Modified on May  6 2025 by ChatGPT
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb

def file_exists_safe(path):
    normalized = os.path.normpath(path.strip())
    if len(normalized) >= 260:
        normalized = r'\\?\{}'.format(normalized)
    return normalized

# — Paths —
predLoc     = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\Segmentation'
gtBaseLoc   = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\datasetForCliniciansGradings\final_clinician_gradings'
saveLocBase = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\SegmentationOverlapWithGT\clinicians_gradings'

# — Pixel spacing (mm² per pixel) —
xyspacing = 0.01174  # mm per pixel
area_per_pixel = xyspacing ** 2

# — Load Dylan (ground-truth) masks & compute GT areas —
gt_folder_dylan = os.path.join(gtBaseLoc, 'dylan')
gt_areas = {}
for fn in os.listdir(gt_folder_dylan):
    if fn.lower().endswith('.tif'):
        p = file_exists_safe(os.path.join(gt_folder_dylan, fn))
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            mask = (m > 0).astype(np.uint8)
            gt_areas[fn] = mask.sum() * area_per_pixel

# — Compute annotation-area differences for each human grader —
graders = ['alisa', 'leon', 'marco', 'mehdi']
area_diff_by_grader = {}

for grader in graders:
    diffs = []
    folder = os.path.join(gtBaseLoc, grader)
    for gt_name, gt_area in gt_areas.items():
        # find the matching annotation file
        for fn in os.listdir(folder):
            if fn.lower().endswith('.png'):
                key = fn.rsplit('__', 1)[-1].replace('.png', '.tif')
                if key == gt_name:
                    p = file_exists_safe(os.path.join(folder, fn))
                    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        mask = (img > 0).astype(np.uint8)
                        annot_area = mask.sum() * area_per_pixel
                        diffs.append(annot_area - gt_area)
                    break
    area_diff_by_grader[grader] = diffs

# — Compute AI-area differences vs GT —
area_diff_ai = []
for gt_name, gt_area in gt_areas.items():
    pred_path = os.path.join(predLoc, gt_name)
    if os.path.exists(pred_path):
        pm = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pm is not None:
            mask = (pm > 0).astype(np.uint8)
            pred_area = mask.sum() * area_per_pixel
            area_diff_ai.append(pred_area - gt_area)

# — Prepare data for plotting —
display_names = {
    'alisa': 'Ophthalmologist 1',
    'leon':  'Ophthalmologist 2',
    'marco': 'Ophthalmologist 3',
    'mehdi': 'Ophthalmologist 4',
    'AI':    'Deep Learning Model'
}

all_vals, all_labels = [], []
for grader, vals in area_diff_by_grader.items():
    all_vals.extend(vals)
    all_labels.extend([display_names[grader]] * len(vals))

all_vals.extend(area_diff_ai)
all_labels.extend([display_names['AI']] * len(area_diff_ai))

# — Plot violin of area differences (Predicted − GT) —
order = ['Deep Learning Model', 'Ophthalmologist 1', 'Ophthalmologist 2', 'Ophthalmologist 3', 'Ophthalmologist 4']

# assemble into a DataFrame for convenience
df_plot = pd.DataFrame({
    'Method': all_labels,
    'AreaDiff': all_vals
})


# — Compute mean bias per ophthalmologist and range —
oph_bias = (
    df_plot[df_plot['Method'].str.contains('Ophthalmologist')]
    .groupby('Method')['AreaDiff']
    .mean()
)
print("Mean biases by Ophthalmologist:")
print(oph_bias)
print(f"Range of biases: {oph_bias.min():.2f} to {oph_bias.max():.2f} mm²")


plt.figure(figsize=(8,6))
sns.violinplot(
    x='Method', 
    y='AreaDiff',
    data=df_plot,
    order=order,
    inner=None     # turn off the default “points” so it’s cleaner underneath
)

# overlay mean ± 95% CI

sns.pointplot(
    x='Method',
    y='AreaDiff',
    data=df_plot,
    order=order,
    estimator=np.mean,  # <— explicitly plot the mean
    join=False,
    ci=95,
    color='k',
    capsize=0.1,
    markers='D',
    err_kws={'linestyle': '--', 'elinewidth': 0.5}
)

plt.axhline(0, linestyle='--', color='gray')
plt.ylabel('Area Difference (Annotations/AI − GT) [mm²]')
plt.title('Violin Plot of Area Differences\n(Annotations/AI vs. Ground Truth)')
plt.tight_layout()

# — Save output —
os.makedirs(saveLocBase, exist_ok=True)
outp = os.path.join(saveLocBase, 'area_diff_violin_annotations_vs_gt_and_ai.tif')
plt.savefig(outp, dpi=600)
plt.close()
