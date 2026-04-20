# Phishing Email Classifier

Course project for ITCS 5154 at UNC Charlotte.
Reproducing the paper: **Fares et al., 2024 - "Machine Learning Approach for Email Phishing Detection"**.

## What it does

Takes email text and predicts whether the email is a phishing attack or a normal legit email. Uses three machine learning models and compares them: **SVM**, **Random Forest**, and **XGBoost**.

---

## Setup (do this once)

### 1. Install Python packages

Open a terminal in the project folder and run:

```
pip install -r requirements.txt
```

### 2. Download the dataset

Go to: https://www.kaggle.com/datasets/subhajournal/phishingemails

Click **Download**. Unzip it. You'll get a file called `Phishing_Email.csv` (~18,600 rows).

Place that file inside the `data/` folder so you have:

```
data/Phishing_Email.csv
```

---

## How to run the project

The project has 3 scripts in the `src/` folder, numbered in the order you run them.

### Script 1 - Train the models

```
python src/1_train_models.py
```

Takes about 2-5 minutes. This reads the Kaggle CSV, trains all 3 models, and saves them. At the end it prints a table of accuracy/precision/recall/F1 for each model.

**Re-run this any time you want to retrain from scratch** (for example if you change the cleaning rules or use a different dataset).

Output files after running:
- `models/svm.joblib`, `random_forest.joblib`, `xgboost.joblib`, `tfidf_vectorizer.joblib`
- `results/metrics.csv` - the numbers in a table
- `results/model_comparison.png` - bar chart comparing the 3 models
- `results/confusion_matrices.png` - confusion matrix per model

### Script 2 - Test with example emails (sanity check)

```
python src/2_test_example_emails.py
```

Classifies 3 hardcoded emails (2 phishing, 1 legit). Takes a second. Good for confirming the trained model loads correctly.

### Script 3 - Classify your own email (interactive demo)

```
python src/3_classify_your_email.py
```

Paste any email into the terminal, type `END` on a new line and press Enter, and the model classifies it. You can keep pasting emails until you type `QUIT`.

**This is the main demo for the video presentation.**

Example session:

```
> python src/3_classify_your_email.py

Loading the trained SVM model...
Model loaded.

=======================================================
  Phishing Email Classifier - Interactive Demo
=======================================================

Paste the email below. When done, type END on its own line.
(Or type QUIT to exit the program.)
-------------------------------------------------------
URGENT! Your PayPal account has been limited.
Click here to verify: http://paypa1-secure.com
END

-------------------------------------------------------
  RESULT: PHISHING
  confidence score: 1.842
  (positive = phishing, negative = legit,
   larger magnitude = more confident)
-------------------------------------------------------
```

---

## Folder layout

```
phishing_project/
├── data/                         put Phishing_Email.csv here
│   └── Phishing_Email.csv
├── models/                       trained models (created by script 1)
│   ├── svm.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── tfidf_vectorizer.joblib
├── results/                      charts and metrics (created by script 1)
│   ├── metrics.csv
│   ├── model_comparison.png
│   └── confusion_matrices.png
├── src/
│   ├── 1_train_models.py
│   ├── 2_test_example_emails.py
│   └── 3_classify_your_email.py
├── README.md
└── requirements.txt
```

---

## My contributions vs. the paper

**From the paper (Fares et al., 2024):**
- Overall pipeline: clean text → TF-IDF → train classifier → evaluate
- Use of SVM and Random Forest
- Evaluation metrics (accuracy, precision, recall, F1)

**My own work:**
- Re-implemented the full pipeline in Python from scratch
- Applied it to a public Kaggle dataset (the paper used a private corpus)
- Added **XGBoost** as a third model for comparison
- Designed the text preprocessing rules (URL / number tokenization)
- Built an interactive classifier (`3_classify_your_email.py`) for the video demo
- Generated the comparison chart and confusion matrices
