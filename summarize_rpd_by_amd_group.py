# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:41:40 2025

@author: mukherjees9
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import pdb

# 1) Load and concatenate both sheets
xlsx_path = r'Z:\Souvick\Projects\Overlap_Labels_With_Original_Image\Unet-Keras-3D\data\membrane\train\MAINSHEET_da_study_2022-04-28.xlsx'
saveLoc = r'Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\results\SegmentationOverlapWithGT\clinicians_gradings'

sheets = pd.read_excel(xlsx_path, sheet_name=['Study', 'Fellow'])
df = pd.concat(sheets.values(), ignore_index=True)

# ────────────── INSERT HERE ──────────────
# 1a) Compute RPD area stats and plot histogram
df['oct_rpd_area'] = pd.to_numeric(df['oct_rpd_area'], errors='coerce')
valid_area = df.loc[df['oct_rpd_area'] < 888, 'oct_rpd_area']

mean_area = valid_area.mean()
std_area  = valid_area.std()
min_area  = valid_area.min()
max_area  = valid_area.max()

print(f"Mean RPD area: {mean_area:.2f} mm² ± {std_area:.2f} mm²")
print(f"Range: {min_area:.2f} – {max_area:.2f} mm²")

plt.figure()
plt.hist(valid_area.dropna(), bins=30)
plt.xlabel("OCT RPD Area (mm²)")
plt.ylabel("Frequency")
plt.title("Distribution of OCT RPD Area")
plt.tight_layout()
plt.savefig(os.path.join(saveLoc, 'RPD_Area_from_RC_GT.tif'), dpi=200)
plt.show()
plt.close('all')
# ────────────── END INSERT ──────────────

# 2) Rename and filter AMD severity column
df = df.rename(columns={'fp_amdsc_corrected-for-cnv': 'amdsc'})
df = df[df['amdsc'].notna()]

# make sure Age is numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# 3) Flags for “definite” RPD
df['fp_rpd_definite']  = df['fp_rpd'].isin([2, 3]).astype(int)
df['oct_rpd_definite'] = (df['oct_rpd'] == 2).astype(int)

# 4) Base demographics + RPD counts
grouped = df.groupby('amdsc').agg(
    n_eyes        = ('amdsc', 'size'),
    mean_age      = ('Age',  'mean'),
    n_male        = ('Sex',  lambda s: (s == 'M').sum()),
    n_female      = ('Sex',  lambda s: (s == 'F').sum()),
    fp_rpd_def_n  = ('fp_rpd_definite',  'sum'),
    oct_rpd_def_n = ('oct_rpd_definite', 'sum'),
    mean_rpd_area = ('oct_rpd_area',  'mean')
)

# 5) Race & Ethnicity counts
race_ct = pd.crosstab(df['amdsc'], df['Race'])
# define your full‐name map
race_map = {
    'B': 'Black / African‑American',
    'C': 'Caucasian (White)',
    'K': 'Asian – Korean',
    'S': 'Asian – South/Southeast Asian',
    'U': 'Unknown / Not reported',
}

# rebuild each column name using the mapped label
race_ct.columns = [
    f"race_{race_map.get(code, code)}"
    for code in race_ct.columns
]
eth_ct  = pd.crosstab(df['amdsc'], df['Ethnicity'])
eth_ct.columns = [f"eth_{c}" for c in eth_ct.columns]

# 6) RPD sub‑features crosstabs
def make_crosstab(col):
    ct = pd.crosstab(df['amdsc'], df[col])
    ct.columns = [f"{col}_{c}" for c in ct.columns]
    return ct

lct_ct = make_crosstab('oct_rpd_lct')
dst_ct = make_crosstab('oct_rpd_dst')
ptt_ct = make_crosstab('oct_rpd_ptt')
ppa_ct = make_crosstab('oct_ppa_rpd')

# 7) Merge all into summary
summary = (
    grouped
    .join(race_ct, how='left')
    .join(eth_ct,  how='left')
    .join(lct_ct,  how='left')
    .join(dst_ct,  how='left')
    .join(ptt_ct,  how='left')
    .join(ppa_ct,  how='left')
    .sort_index()
)

# 8) Fill NaNs with 0
summary = summary.fillna(0)

# 9) Convert counts to percentages (except n_eyes & mean_age)
# after you have your `summary` DataFrame

# 1) isolate the count columns (everything except n_eyes, mean_age, fp_rpd_def_n, oct_rpd_def_n)
counts = summary.drop(columns=['n_eyes','mean_age','fp_rpd_def_n','oct_rpd_def_n', 'mean_rpd_area'])

# 2) identify which columns are your RPD sub‑features
subfeatures = (
    list(filter(lambda c: c.startswith('oct_rpd_lct_'), counts.columns)) +
    list(filter(lambda c: c.startswith('oct_rpd_dst_'), counts.columns)) +
    list(filter(lambda c: c.startswith('oct_rpd_ptt_'), counts.columns)) +
    list(filter(lambda c: c.startswith('oct_ppa_rpd_'), counts.columns))
)

# 3) build a pct DataFrame
pct = pd.DataFrame(index=summary.index)
for col in counts.columns:
    if col in subfeatures:
        denom = summary['oct_rpd_def_n']
    else:
        denom = summary['n_eyes']
    pct[col] = (counts[col] / denom * 100).round(1)

# 4) recombine with your key columns
summary_pct = pd.concat([
    summary[['n_eyes','mean_age','fp_rpd_def_n','oct_rpd_def_n', 'mean_rpd_area']],
    pct
], axis=1)

# 5) (optional) save out
summary_pct.to_excel(os.path.join(saveLoc, 'RPD_by_AMDgroup_summary_pct.xlsx'))


