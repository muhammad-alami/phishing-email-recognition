# Phishing Email Classifier

Final project for ITCS 5154 at UNC Charlotte.
Reproducing Fares et al., 2024 — "Machine Learning Approach for Email Phishing Detection."

## What this does

Reads an email and predicts if it's phishing or legitimate. Trains three models (SVM, Random Forest, XGBoost) on TF-IDF features and compares them. SVM wins.

## Setup
pip install -r requirements.txt

Download the dataset from https://www.kaggle.com/datasets/subhajournal/phishingemails and place `Phishing_Email.csv` in the `data/` folder.

## Run the pipeline
python src/data_prep.py
python src/train_svm.py
python src/train_rf.py
python src/train_xgb.py
python src/evaluate.py
python src/make_figures.py

Whole thing runs in under two minutes on a laptop.

## Try the interactive demo
python src/classify_email.py

Paste an email, type `END` on a new line, get a label and confidence score. Type `QUIT` to exit.
