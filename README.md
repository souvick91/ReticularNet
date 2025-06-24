# ReticularNet

ReticularNet is a Python-based pipeline for segmenting Reticular Pseudodrusen (RPD) in fundus images using a deep learning model and evaluating its performance against manual clinician gradings. This repository provides scripts to compute segmentation metrics, perform statistical analyses, generate visualizations, and summarize clinical characteristics.

## Features

* **Segmentation Metrics**: Compute Dice similarity coefficients (pairwise, majority vote, full agreement).
* **Reliability Analysis**: Calculate intraclass correlation (ICC) and perform Wilcoxon and Wald tests.
* **Visualizations**: Generate violin plots, Bland–Altman plots, scatter/regression plots, ROC curves.
* **Clinical Summaries**: Produce Table 1 characteristics summaries (total/train/test by RPD+/RPD–, nearest visit within ±90 days).
* **Detection Performance**: Compute ROC‐AUC, optimal Youden’s J thresholds, and Dice at threshold for AI and human graders.

## Repository Structure

```
ReticularNet/python_scripts/
├── compute_dice_statistics.py           # Pairwise & consensus Dice analysis
├── compute_agreement_metrics.py         # ICC & paired Wilcoxon tests
├── segmentation_performance_visualization.py  # Overlays + DSC, area plots, Bland-Altman
├── area_difference_violin_plot.py       # Violin plot of area differences (human vs AI)
├── compute_segmentation_metrics.py      # Lesion counts, area, contour area + concave-hull overlays
├── compare_lesion_metrics_to_ground_truth.py  # Cluster-robust signed differences & Wald tests
├── generate_table_characteristics_summary.py # Table: clinical characteristics by RPD status
├── summarize_rpd_by_amd_group.py        # RPD prevalence & demographics by AMD severity group
├── compute_detection_metrics.py         # ROC, Youden’s J, AUC, p-values, Dice at threshold
├── compute_roc_auc_se_and_pvalues.py    # AUC ± SE vs. chance for AI, graders, combined
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/ReticularNet.git
   cd ReticularNet
   ```
2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

* Each script defines file paths at the top (e.g., `gt_folder`, `pred_folder`, `saveLoc`).
* Update these variables to point to your local directories containing ground-truth masks, AI predictions, raw IR images, and desired output locations.

## Usage

Run the desired analysis script from the command line. Examples:

```bash
python compute_dice_statistics.py
python compute_agreement_metrics.py
python area_difference_violin_plot.py
```

Results (Excel, CSV, plots) will be saved to the output directories specified in each script.

## License

This project is licensed under the Apache License 2.0. See LICENSE for details.

## Contact

Questions or feedback? Please open an issue or contact the maintainer:

Souvick Mukherjee
(mailto:souvick25031991@gmail.com)
