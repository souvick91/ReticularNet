#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare each grader (4 clinicians + AI Model) vs. ground truth
for lesion count, contour area, and total area.
Per-grader mean signed difference and Wald-t p-value are saved,
with cluster-robust standard errors by image.
"""

import pandas as pd
import statsmodels.formula.api as smf
import os

def analyze_metric(df, diff_col, label):
    """
    Fit Diff ~ C(grader) - 1 with cluster-robust SE on image.
    Returns a DataFrame with one row per grader:
      - mean_diff: coefficient for that grader (mean signed difference)
      - p_value: Wald t-test p-value testing coef = 0
    """
    # drop any rows missing the key columns
    sub = df.dropna(subset=['image', 'grader', diff_col]).copy()
    # fit with no intercept so each grader has its own coefficient
    model = smf.ols(f"{diff_col} ~ C(grader) - 1", data=sub).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["image"]}
    )

    rows = []
    for term in model.params.index:
        # term looks like "C(grader)[Alisa]" etc.
        grader = term.split('[')[1].rstrip(']')
        rows.append({
            "metric":    label,
            "grader":    grader,
            "mean_diff": model.params[term],
            "p_value":   model.pvalues[term]
        })
    return pd.DataFrame(rows)

def main():
    # 1) Load your data
    inp = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations" \
          r"\results\SegmentationOverlapWithGT\clinicians_gradings" \
          r"\metrics_vs_groundTruth.xlsx"
    df = pd.read_excel(inp, dtype={"grader": str})

    # 2) Ensure the image column is all strings (no mixed float/string)
    df = df.dropna(subset=["image"])
    df["image"] = df["image"].astype(str)

    # 3) Compute signed differences vs. ground truth
    df["CountDiff"]       = df["grader_lesion_count"]     - df["gt_lesion_count"]
    df["ContourAreaDiff"] = df["grader_contour_area_mm2"] - df["gt_contour_area_mm2"]
    df["TotalAreaDiff"]   = df["grader_total_area_mm2"]   - df["gt_total_area_mm2"]

    # 4) Analyze each metric
    metrics = [
        ("CountDiff",       "lesion_count_diff"),
        ("ContourAreaDiff", "contour_area_diff"),
        ("TotalAreaDiff",   "total_area_diff"),
    ]
    all_summaries = []
    for col, label in metrics:
        summ = analyze_metric(df, col, label)
        all_summaries.append(summ)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    # reorder columns for clarity
    summary_df = summary_df[['metric', 'grader', 'mean_diff', 'p_value']]

    # 5) Save to Excel
    out_dir = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations" \
              r"\results\SegmentationOverlapWithGT\clinicians_gradings" \
              r"\lesionCount_contourArea_AIVsHuman"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "diff_vs_GT_per_grader.xlsx")
    summary_df.to_excel(out_path, index=False)

    print(f"Written per-grader difference summary to:\n  {out_path}")
    print("\nPreview:")
    print(summary_df)

if __name__ == "__main__":
    main()
