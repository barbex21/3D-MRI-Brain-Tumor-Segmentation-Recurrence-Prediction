# 3D MRI Brain Tumor Segmentation & Recurrence Prediction

## 📌 Overview
This repository contains a deep learning pipeline for 3D MRI brain tumor segmentation, developed as part of a health data science research initiative at **IIS La Fe (GIBI230 Research Group)** in Valencia, Spain.

The primary research objective is to **systematically evaluate different MRI modalities and their combinations** to determine the optimal configuration for training a robust 3D U-Net model. By identifying the most effective sequences for segmenting specific tumor sub-regions, the pipeline ultimately aims to support the localization and prediction of tumor recurrence sites.

## 🔬 Research Methodology & Objectives
Rather than relying on a fixed, predefined input, this project explores the independent and combined predictive power of various MRI sequences:
* **Modalities Investigated:** Native T1, T1ce (Contrast-Enhanced), T2, and FLAIR.
* **Target Regions:** Enhancing Tumor (`et`), Necrotic Core (`necr`), and their combinations.
* **Evaluation Strategy:** 
  1. **Single-Modality Baselines:** Training isolated models to map specific modalities to specific targets (e.g., assessing T1 for `et`, T1 for `necr`, T1ce for `et`, etc.).
  2. **Comparative Analysis:** Evaluating which individual slices yield the highest clinical accuracy and mitigate false positives (especially in challenging postoperative scenarios).
  3. **Multi-Modal Fusion:** Developing optimal combination strategies based on the best-performing individual sequences to maximize spatial and contextual feature extraction.
 
## 💾 Dataset: RHUH-GBM
This project utilizes the publicly available **RHUH-GBM** (Rennes University Hospital Glioblastoma) dataset from The Cancer Imaging Archive (TCIA).

* **Source:** [RHUH-GBM Collection on TCIA](https://www.cancerimagingarchive.net/collection/rhuh-gbm/)
* **Data Preparation:** To run this pipeline, download the dataset and place it in the `data/input/dataset` directory. The `01_dataset_index.py` script is designed to automatically parse the patient folders, extract the required NIfTI volumes, and generate the necessary training/validation/test splits.

## 🛠️ Tech Stack & Architecture
* **Frameworks:** PyTorch, PyTorch Lightning, MONAI
* **Architecture:** 3D U-Net (tailored for volumetric medical data)
* **Data Processing:** OpenCV, robust spatial augmentations (RandAffined, RandGaussianNoised)
* **Optimization:** AMP (Automatic Mixed Precision), DiceCELoss

## 📊 Clinical Insights & Preliminary Results

### Phase 1: Native T1 Baseline Evaluation (Completed)
The initial baseline evaluated the predictive capacity of **native T1 modalities** strictly for segmenting the enhancing tumor (`et`) core.

| Modality | Target | Val Dice | Test Dice | Precision | Recall | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| T1 Only | `et` | 0.1328 | 0.1260 | 0.0907 | 0.3682 | Baseline model |

**Clinical Analysis:** The model was tasked with identifying the enhancing tumor core without access to contrast-enhanced (T1ce) scans. Phase-wise evaluation revealed significant struggles with postoperative scans (Postop Test Dice: 0.0104) and a high rate of false positives in empty cases, indicating that native T1 contrast alone is insufficient to differentiate between active tumor tissue and postoperative surgical cavities or edema.

### Phase 2: Modality Expansion (Ongoing)
Subsequent experiments are currently running to evaluate:
* `T1` mapping to `necr`
* `T1ce` mapping to `et` and `necr`
* `T2` and `FLAIR` independent evaluations
* Optimal multi-modal fusion strategies (e.g., T1ce + T2 + FLAIR) to resolve the high false-positive rates observed in the baseline.

## 📂 Production-Ready Repository Structure

```text
mri-tumor-segmentation-lafe/
├── config/                 # YAML configuration files for modality/target sweeps
├── data/                   # Data loaders (raw datasets ignored via .gitignore)
├── src/                    
│   ├── dataset.py          # MONAI transforms and multi-modal pre-processing
│   ├── model.py            # 3D U-Net architecture definition
│   ├── train.py            # Training loop with AMP and GradScaler
│   ├── evaluate.py         # Sliding window inference & metric calculation
│   └── utils.py            # Helper functions
├── notebooks/              # EDA and experimental visual overlays
├── Dockerfile              # Containerization configuration
├── requirements.txt        # Exact dependencies
└── README.md
