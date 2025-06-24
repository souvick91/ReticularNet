#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute agreement metrics and p-values for AI model and graders
against ground truth, then save summary to Excel.
"""

import pdb
import os
import pandas as pd
import pingouin as pg
import pdb
from scipy.stats import wilcoxon

# Compute ICC(2,1) for agreement on each metric
def compute_icc(true_vals, pred_vals):
    df_icc = pd.concat([
        pd.DataFrame({'Subject': sub['image'], 'Rater': 'True', 'Score': true_vals}),
        pd.DataFrame({'Subject': sub['image'], 'Rater': 'Pred', 'Score': pred_vals})
    ], ignore_index=True)
    icc_df = pg.intraclass_corr(
        data=df_icc,
        targets='Subject',
        raters='Rater',
        ratings='Score'
    )
    # pick the ICC2 row
    return icc_df.query("Type=='ICC2'")['ICC'].values[0]

# ─── USER PATHS ────────────────────────────────────────────────────────────────
input_path = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\SegmentationOverlapWithGT\clinicians_gradings\metrics_vs_groundTruth.xlsx"
output_path = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\SegmentationOverlapWithGT\clinicians_gradings\graded_model_comparison.xlsx"
# ────────────────────────────────────────────────────────────────────────────────

# Load the metrics sheet
df = pd.read_excel(input_path)

# Prepare storage for summary
summary = []

 # Only loop through real graders/models, skipping NaN
methods = df['grader'].dropna().unique()
for method in methods:
    sub = df[df['grader'] == method]

    # Calculate Dice mean and SD
    mean_dsc = sub['dice_vs_ground_truth'].mean()
    sd_dsc   = sub['dice_vs_ground_truth'].std()

    # Calculate raw lesion‐count mean±SD for GT and method
    mean_gt_count   = sub['gt_lesion_count'].mean()
    sd_gt_count     = sub['gt_lesion_count'].std()
    mean_md_count   = sub['grader_lesion_count'].mean()
    sd_md_count     = sub['grader_lesion_count'].std()
    
    # Calculate raw pixel‐area mean±SD for GT and method
    mean_gt_pix     = sub['gt_total_area_mm2'].mean()
    sd_gt_pix       = sub['gt_total_area_mm2'].std()
    mean_md_pix     = sub['grader_total_area_mm2'].mean()
    sd_md_pix       = sub['grader_total_area_mm2'].std()
    
    # Calculate raw contour‐area mean±SD for GT and method
    mean_gt_cont    = sub['gt_contour_area_mm2'].mean()
    sd_gt_cont      = sub['gt_contour_area_mm2'].std()
    mean_md_cont    = sub['grader_contour_area_mm2'].mean()
    sd_md_cont      = sub['grader_contour_area_mm2'].std()

    icc_count   = compute_icc(sub['gt_lesion_count'],    sub['grader_lesion_count'])
    icc_pixarea = compute_icc(sub['gt_total_area_mm2'],  sub['grader_total_area_mm2'])
    icc_contour = compute_icc(sub['gt_contour_area_mm2'],sub['grader_contour_area_mm2'])

    # Paired Wilcoxon signed-rank tests
    _, p_count   = wilcoxon(sub['gt_lesion_count'], sub['grader_lesion_count'])
    _, p_pixarea = wilcoxon(sub['gt_total_area_mm2'], sub['grader_total_area_mm2'])
    _, p_contour = wilcoxon(sub['gt_contour_area_mm2'], sub['grader_contour_area_mm2'])

    summary.append({
        'Grader': method,
        'DSC (mean±SD)':   f"{mean_dsc:.2f}±{sd_dsc:.2f}",

        # Raw lesion‐count
        'GT Lesion Count (mean±SD)':    f"{mean_gt_count:.1f}±{sd_gt_count:.1f}",
        'Method Lesion Count (mean±SD)':f"{mean_md_count:.1f}±{sd_md_count:.1f}",
        
        # Raw pixel‐area
        'GT Pixel Area (mean±SD)':      f"{mean_gt_pix:.2f}±{sd_gt_pix:.2f}",
        'Method Pixel Area (mean±SD)':  f"{mean_md_pix:.2f}±{sd_md_pix:.2f}",
        
        # Raw contour‐area
        'GT Contour Area (mean±SD)':    f"{mean_gt_cont:.2f}±{sd_gt_cont:.2f}",
        'Method Contour Area (mean±SD)':f"{mean_md_cont:.2f}±{sd_md_cont:.2f}",
        
        'Lesion Count ICC':   round(icc_count, 2),
        'Lesion Count p':      p_count,
        'Pixel Area ICC':   round(icc_pixarea, 2),
        'Pixel Area p':      p_pixarea,
        'Contour Area ICC': round(icc_contour, 2),
        'Contour Area p':    p_contour
    })

# Build a summary DataFrame
summary_df = pd.DataFrame(summary)

# --- begin: per-grader Dice comparison AI vs human ---
# pivot so rows are images, columns are graders + AI
# --- new code: drop any rows missing a dice score, then pivot_table with aggfunc='first' ---
df_scores = (
    df
    .dropna(subset=['image', 'grader', 'dice_vs_ground_truth'])
    .drop_duplicates(subset=['image', 'grader'], keep='first')
)
dice_wide = df_scores.pivot_table(
    index='image',
    columns='grader',
    values='dice_vs_ground_truth',
    aggfunc='first'
)

# isolate AI column and list of grader columns
ai_scores   = dice_wide['AI Model']
grader_cols = [c for c in dice_wide.columns if c != 'AI Model']

w_dict = {}
p_dict = {}
# loop over each grader
for grader in grader_cols:
    grader_scores = dice_wide[grader]
    # only keep images with both AI & this grader
    mask = ai_scores.notna() & grader_scores.notna()
    ai_vals     = ai_scores[mask]
    grader_vals = grader_scores[mask]
    # paired Wilcoxon
    stat, p = wilcoxon(ai_vals, grader_vals)
    w_dict[grader] = stat
    p_dict[grader] = p    
    print(f"AI vs {grader} Dice: W={stat:.2f}, p={p:.3e}")
# --- end: per-grader Dice comparison ---

# map into summary_df (AI row will get NaN)
summary_df['Dice W'] = summary_df['Grader'].map(w_dict)
summary_df['Dice p'] = summary_df['Grader'].map(p_dict)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save to Excel
summary_df.to_excel(output_path, index=False)
print(f"Saved comparison summary → {output_path}")
