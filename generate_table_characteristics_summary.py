#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate “Table 1” characteristics summary for Total, Training and Testing,
each split into RPD+ vs RPD–, by matching clinic dates within ±90 days of image dates
(and picking only the nearest match).
"""

import os
import glob
import pandas as pd
import pdb

# ─── USER PATHS ────────────────────────────────────────────────────────────────
xlsx_path  = (
    r"Z:\Souvick\Projects\Overlap_Labels_With_Original_Image"
    r"\Unet-Keras-3D\data\membrane\train\MAINSHEET_da_study_2022-04-28.xlsx"
)
save_dir   = (
    r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations"
    r"\results\SegmentationOverlapWithGT\clinicians_gradings\Table 1"
)
train_dir  = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\TrainMasks"
test_dir   = r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\TestMasks"
all_rpd_dir= r"Z:\Souvick\Projects\RPDSegmentation\Manual_Masked_Segmentations\allImages"
os.makedirs(save_dir, exist_ok=True)

# ─── 1) Load & prepare clinical data ────────────────────────────────────────────
sheets = pd.read_excel(xlsx_path, sheet_name=['Study','Fellow'])
df = pd.concat(sheets.values(), ignore_index=True)
# After concatenating your sheets:
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'], errors='coerce')

# Adjust these to your real column names:
df['PATID']      = df['PATID'].astype(int)                      # patient ID
df['Eye']        = df['Eye'].astype(str)                        # "OD"/"OS"
df['visit_date'] = pd.to_datetime(df['Visit_Date'])            # exam date

# flags & numeric conversions
df['fp_rpd_definite']  = df['fp_rpd'].isin([2,3]).astype(int)
df['oct_rpd_definite'] = (df['oct_rpd'] == 2).astype(int)
df['oct_rpd_area']     = pd.to_numeric(df['oct_rpd_area'], errors='coerce')

# ─── 2) Gather TIFF filenames ─────────────────────────────────────────────────
def list_tifs(d):
    return [os.path.basename(p) for p in glob.glob(os.path.join(d, '*.tif'))]

train_files   = list_tifs(train_dir)
test_files    = list_tifs(test_dir)
total_files   = train_files + test_files
rpd_plus_set  = set(list_tifs(all_rpd_dir))

# ─── 3) Summary helper ─────────────────────────────────────────────────────────
race_map = {
    'B': 'Black / African-American',
    'C': 'Caucasian (White)',
    'K': 'Asian – Korean',
    'S': 'Asian – South/Southeast Asian',
    'U': 'Unknown / Not reported'
}
# ─── Build dynamic subfeature dict ──────────────────────────────────────────────
subfeat_cols = ['oct_rpd_lct', 'oct_rpd_dst', 'oct_rpd_ptt', 'oct_ppa_rpd']

# For each of those columns, get the sorted list of unique, non-null values
subfeats = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in subfeat_cols
}

# ─── Build dynamic ethnicity list ──────────────────────────────────────────────
# (you can do the same for race if you like)
eth_cats = sorted(df['Ethnicity']
                    .dropna()
                    .unique()
                    .tolist())

def compute_summary(file_list):
    # collect the index of the nearest match for each image (if any)
    matched_idxs = []
    mismatches   = [] 
    for fn in file_list:
        base = os.path.splitext(fn)[0]
        parts = base.split('_')
        if len(parts) != 5:
            continue
        pid_str, _, _, date_str, eye = parts
        try:
            pid = int(pid_str)
            img_date = pd.to_datetime(date_str, format='%Y%m%d')
        except ValueError:
            continue

        # filter ±90 days & matching eye/patid
        mask = (
            (df['PATID'] == pid) &
            (df['This_Data_Corresponds_To2']   == eye) &
            (df['Visit_Date'] >= img_date - pd.Timedelta(days=90)) &
            (df['Visit_Date'] <= img_date + pd.Timedelta(days=90))
        )
        sub = df.loc[mask]
        if sub.empty:
            mismatches.append(fn)
            continue

        # pick the single nearest date
        deltas = (sub['visit_date'] - img_date).abs()
        nearest_idx = deltas.idxmin()
        matched_idxs.append(nearest_idx)

    # get a DataFrame of all matched rows (one per image)
    n_matched_files = len(matched_idxs)
    sub_df = df.loc[matched_idxs].drop_duplicates()
    # DEBUG
    print(
        f"[DEBUG] {len(file_list)} files → "
        f"{n_matched_files} matched file entries → "
        f"{len(sub_df)} unique clinical rows → "
        f"{len(mismatches)} mismatches"
    )

    # compute all stats on sub_df
    n_eyes        = len(sub_df)
    mean_age      = sub_df['Age'].mean()
    fp_n          = sub_df['fp_rpd_definite'].sum()
    oct_n         = sub_df['oct_rpd_definite'].sum()
    mean_area     = sub_df['oct_rpd_area'].mean()
    n_male        = (sub_df['Sex']=='M').sum()
    n_female      = (sub_df['Sex']=='F').sum()

    # race/ethnicity %
    race_pct = (sub_df['Race'].map(race_map)
                      .value_counts(normalize=True) * 100).round(1)
    eth_pct  = (sub_df['Ethnicity']
                      .value_counts(normalize=True) * 100).round(1)

    stats = {
        'n_eyes':        n_eyes,
        'mean_age':      round(mean_age,6),
        'fp_rpd_def_n':  fp_n,
        'oct_rpd_def_n': oct_n,
        'mean_rpd_area': round(mean_area,3),
        'n_male':        n_male,
        'n_female':      n_female,
    }

    # add race & ethnicity columns
    for full in race_map.values():
        stats[f"race_{full}"] = race_pct.get(full, 0.0)
    for e in eth_cats:
        stats[f"eth_{e}"]    = eth_pct.get(e, 0.0)

    # sub‐feature % out of oct_n (or 1 if zero)
    denom = oct_n or 1
    for col, cats in subfeats.items():
        ct = sub_df[col].value_counts()
        for cat in cats:
            stats[f"{col}_{cat}"] = round((ct.get(cat,0)/denom)*100,1)
    
    # ethnicity % (instead of a hard-coded list)
    eth_pct = (sub_df['Ethnicity']
                .value_counts(normalize=True) * 100).round(1)
    for e in eth_cats:
        stats[f"eth_{e}"] = eth_pct.get(e, 0.0)


    return stats, mismatches

# ─── 4) Build the final DataFrame ──────────────────────────────────────────────
records = []
groups = {
    'Total':    total_files,
    'Training': train_files,
    'Testing':  test_files,
}
total_mismatches = []

for grp_name, flist in groups.items():
    for status, test_fn in [('RPD+', lambda f: f in rpd_plus_set),
                            ('RPD-', lambda f: f not in rpd_plus_set)]:
        subset = [f for f in flist if test_fn(f)]
        stats, mismatches = compute_summary(subset)
        
        if grp_name == 'Total':
            total_mismatches.extend(mismatches)

        stats['Group']      = grp_name
        stats['RPD_Status'] = status
        records.append(stats)

summary_df = pd.DataFrame(records)
# reorder columns
cols = ['Group','RPD_Status'] + [c for c in summary_df.columns
                                 if c not in ('Group','RPD_Status')]
summary_df = summary_df[cols]

# ─── Incorporate missing‐visit Age/Gender into each RPD– subgroup ─────────
missing_path = (
    r"Z:\Souvick\Projects\Overlap_Labels_With_Original_Image"
    r"\Unet-Keras-3D\data\membrane\train"
    r"\RPDMinusPaperMissingRCVisitsSexAge.xlsx"
)
if os.path.exists(missing_path):
    miss_df = pd.read_excel(missing_path)
    n_miss     = len(miss_df)
    sum_age    = miss_df['Age'].sum()
    male_miss  = (miss_df['Gender']=='M').sum()
    female_miss= (miss_df['Gender']=='F').sum()

    for grp in ['Total','Training']:
        mask = (summary_df['Group']==grp) & (summary_df['RPD_Status']=='RPD-')
        old_n      = summary_df.loc[mask, 'n_eyes'].iat[0]
        old_age    = summary_df.loc[mask, 'mean_age'].iat[0]
        old_m      = summary_df.loc[mask, 'n_male'].iat[0]
        old_f      = summary_df.loc[mask, 'n_female'].iat[0]

        new_n      = old_n + n_miss
        # weighted mean age across old + new
        new_age    = (old_age*old_n + sum_age) / new_n
        new_m      = old_m + male_miss
        new_f      = old_f + female_miss

        summary_df.loc[mask, 'n_eyes']   = new_n
        summary_df.loc[mask, 'mean_age'] = new_age
        summary_df.loc[mask, 'n_male']   = new_m
        summary_df.loc[mask, 'n_female'] = new_f
        print(f"Updated {grp}/RPD–: +{n_miss} visits → age/sex")
else:
    print(f"Missing‐visits file not found: {missing_path}")



# ─── 5) Save to Excel ──────────────────────────────────────────────────────────
out_path = os.path.join(save_dir,
    'Table1_total_train_test_by_RPD_tol90d_nearest.xlsx')
summary_df.to_excel(out_path, index=False)
print(f"Saved → {out_path}")

# parse the filenames into fields
records = []
for fn in total_mismatches:
    base = os.path.splitext(fn)[0]
    parts = base.split('_')
    if len(parts) == 5:
        pid_str, _, _, date_str, eye = parts
        records.append({
            'filename': fn,
            'PATID': pid_str,
            'VisitDate': pd.to_datetime(date_str, format='%Y%m%d'),
            'Eye': eye
        })
if records:
    mismatch_df = pd.DataFrame(records)
    mm_path = os.path.join(save_dir, 'Total_group_mismatches.xlsx')
    mismatch_df.to_excel(mm_path, index=False)
    print(f"Saved mismatches → {mm_path}")
else:
    print("No mismatches found for Total group")
    