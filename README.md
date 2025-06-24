# ReticularNet

Automated pixel-level segmentation of Reticular Pseudodrusen (RPD) in infra-red reflectance images by deep learning, with extensive performance evaluation and comparison to clinician gradings.

## Features

* **Deep Learning Model**: Training and testing scripts for RPD segmentation using DeepLabv3+ in MATLAB.
* **Python Analysis Pipeline**: Compute segmentation metrics, agreement statistics, lesion/detection performance, and generate visualizations.
* **Clinical Summaries**: Automated generation of Table characteristics and RPD summaries by AMD group.
* **Statistical Tests**: Intraclass correlation, Wilcoxon, Wald-type tests, ROC/AUC analysis with Youden’s J thresholds.
* **Visualizations**: Violin plots, Bland–Altman plots, scatter/regression, ROC curves.

## Repository Structure

```
ReticularNet/
├── combinedCrossEntropyDiceLoss.m        # Custom loss combining cross-entropy & Dice
├── createLGraphUsingConnections.m        # Build DeepLabv3+ layer graph with skip connections
├── DeepLabv3_Test_for_RPD.m              # Inference script: test RPD segmentation
├── DeepLabv3_Train_for_RPD.m             # Training script: train DeepLabv3+ on IR images
├── DeepLabv3plusResnet18CamVid_v2.mat    # Pretrained checkpoint for transfer learning. Download checkpoints from our [Release v1.0-checkpoints](https://github.com/souvick91/ReticularNet/releases/tag/checkpoints).
├── freezeWeights.m                       # Utility to freeze specified network layers
├── masterTrainTest.m                     # End-to-end train/test workflow wrapper
├── net_finetuned_for_rpd.mat             # Fine-tuned model checkpoint for RPD. Download checkpoints from our [Release v1.0-checkpoints](https://github.com/souvick91/ReticularNet/releases/tag/checkpoints).
├── compute_dice_statistics.py            # Pairwise & consensus Dice analysis
├── compute_agreement_metrics.py          # ICC & paired Wilcoxon tests
├── compute_segmentation_metrics.py       # Lesion counts, areas, contour areas + overlays
├── segmentation_performance_visualization.py  # Overlay figures, scatter/regression, Bland–Altman, violin
├── area_difference_violin_plot.py        # Violin plot of area differences (clinicians vs. AI)
├── compare_lesion_metrics_to_ground_truth.py  # Cluster-robust signed differences & Wald tests
├── generate_table1_characteristics_summary.py # Table 1 summary for total/train/test by RPD status
├── summarize_rpd_by_amd_group.py         # RPD prevalence & demographics by AMD severity group
├── compute_detection_metrics.py          # ROC/AUC, Youden’s J, p-values, Dice at threshold for detection
├── compute_roc_auc_se_and_pvalues.py     # AUC ± SE and p-value vs. chance for AI, graders, combined
├── requirements.yml                      # Python dependencies
├── LICENSE                               # Apache 2.0 License
└── README.md                             # This file
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/ReticularNet.git
   cd ReticularNet
   ```
2. **MATLAB**

   * Ensure you have MATLAB R2023b or later with Deep Learning Toolbox.
   * Add script location to your MATLAB path.
3. **Python**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\\Scripts\\activate
   conda env create -f requirements.yml
   conda activate required_env
   ```

## Configuration

* Modify file paths at the top of each script:

  * **MATLAB**: `DeepLabv3_Train_for_RPD.m` and test scripts expect folders for IR images and ground truth masks.
  * **Python**: Each script defines variables like `gt_folder`, `pred_folder`, and `saveLoc`.

## Usage

### MATLAB Training & Testing

Launch MATLAB and run:

```matlab
% Train model
DeepLabv3_Train_for_RPD
% Test model
DeepLabv3_Test_for_RPD
```

### Python Analyses

From the repo root, run any desired analysis:

```bash
python compute_dice_statistics.py
python compute_agreement_metrics.py
python compute_detection_metrics.py
python segmentation_performance_visualization.py
```

All outputs (Excel reports, CSVs, and plots) are saved to the directories specified in each script.

## Examples

* **Dice statistics**: `compute_dice_statistics.py` produces an Excel file with mean±SD Dice for each pair of graders and consensus masks.
* **ROC curves**: `compute_detection_metrics.py` saves `roc_curves_all_methods.tif` and detection metrics Excel.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contact

For questions or contributions, please open an issue or contact:

Souvick Mukherjee
National Eye Institute, NIH
Email: [souvick25031991@gmail.com](mailto:souvick25031991@gmail.com)
