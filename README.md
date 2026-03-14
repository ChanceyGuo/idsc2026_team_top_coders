# IDSC2026 Team Top Coders

This project is a lightweight and reproducible ECG classification pipeline built on the Brugada-HUCA dataset from PhysioNet.

The project is designed to run mainly in Kaggle. GitHub is used to store scripts, project structure, documentation, and demo files.

## Project team

- Guo Qianqi (ChanceyGuo) : ChanceyGuo@outlook.com
- Jin Yanran : yariena0104@163.com
- Wang Yijie : wangyijie072022@gmail.com

# Project goal

Our goal is to build a simple screening-oriented baseline for distinguishing Brugada-related ECG patterns from normal controls. The focus of this Stage 1 version is reproducibility, interpretability, and a clear pipeline.

# Recommended environment

This project is intended to run in Kaggle because the dataset is already available there and the workflow is easier to reproduce.

# Recommended Kaggle dataset root

`/kaggle/input/datasets/chanceyguo/idsc2026-brugada-huca-raw`

# Repository structure

# data
dataset description and citation notes

# models
trained model files

# notebooks
main Kaggle notebook

# outputs
generated results, figures, and splits

# scripts
main pipeline scripts

# streamlit
demo app

# Main pipeline

- `00_check_environment.py`
- `01_load_data.py`
- `02_make_splits.py`
- `03_explore_data.py`
- `04_visualize_signals.py`
- `05_preprocess_signal.py`
- `06_feature_engineering.py`
- `07_train_model.py`
- `08_evaluate_model.py`

# Label setting

This project uses binary classification.

 `brugada = 0` means Normal
 `brugada > 0` means Brugada

# Preprocessing

We use a simple preprocessing pipeline:

- validate signal shape
- transpose to lead-first format
- apply lead-wise z-score normalization
- apply simple clipping to reduce extreme outliers

# Feature engineering

We extract four simple statistics from each of the 12 leads:

- mean
- standard deviation
- minimum
- maximum

This gives a 48-dimensional feature vector for each subject.

# Models

Current baseline models:

- `lr_unweighted`
- `lr_balanced`
- `rf`

The main model used for reporting is `lr_balanced` because it provides the best balance between recall and overall screening-oriented performance.

# Interpretability

To support interpretability, we export model coefficient and feature importance files in `outputs/results`.

The main interpretability file used for reporting is:

`lr_coefficients_balanced.csv`

# Demo

A simple demo app is provided in `streamlit/app.py`.

It loads the trained model, reads a patient ECG, and shows prediction results with a signal preview.

# Important note

This project is a research prototype for screening-oriented analysis only.

It is not a clinical diagnostic system.

# Dataset

Official dataset page:  
https://physionet.org/content/brugada-huca/1.0.0/

# Citation

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

# 