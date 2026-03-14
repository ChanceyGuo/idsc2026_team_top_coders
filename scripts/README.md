scripts

This folder contains the main executable pipeline scripts.

Pipeline order

- 00_check_environment.py
- 01_load_data.py
- 02_make_splits.py
- 03_explore_data.py
- 04_visualize_signals.py
- 05_preprocess_signal.py
- 06_feature_engineering.py
- 07_train_model.py
- 08_evaluate_model.py

Each script is designed to perform one clear task so that the workflow is easier to reproduce, debug, and review.

The scripts are mainly called from the main Kaggle notebook.
