streamlit

This folder contains the demo app for this project.

Main file:
app.py

Purpose

The app provides a simple interactive interface for loading one patient ECG, extracting features, running the trained model, and showing the prediction result.

Recommended environment

The app is easiest to run on a local machine after the model and dataset are prepared.

Required packages:
streamlit
wfdb
joblib
pandas
numpy
matplotlib

Before running the app, make sure:
the dataset has been extracted into the expected folder structure
the trained model file exists in the models folder
the default data path and model path in app.py match your local setup

Run command

python -m streamlit run streamlit/app.py

Important note

This demo is for research and educational use only.
It is not a clinical diagnostic tool.
