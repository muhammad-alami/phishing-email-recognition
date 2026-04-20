# trains the phishing classifier on the kaggle dataset
# run this first before anything else

import os
import re
import string
import sys
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


DATA_FILE = "data/Phishing_Email.csv"
models_dir = "models"
results_dir = "results"
TEST_SIZE = 0.2
RANDOM_SEED = 42
MIN_EXPECTED_ROWS = 5000  # real dataset has ~18600 rows

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

if not os.path.exists(DATA_FILE):
    print("can't find", DATA_FILE)
    print("download from: https://www.kaggle.com/datasets/subhajournal/phishingemails")
    sys.exit(1)


print("Step 1: Loading the dataset...")

df = pd.read_csv(DATA_FILE)

# drop unnamed index col if pandas added one
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.rename(columns={"Email Text": "text", "Email Type": "email_type"})
df = df.dropna(subset=["text", "email_type"])

# 1 = phishing, 0 = legit
df["label"] = df["email_type"].apply(
    lambda x: 1 if x.strip().lower() == "phishing email" else 0
)

print("  total emails:   ", len(df))
print("  legit emails:   ", (df["label"] == 0).sum())
print("  phishing emails:", (df["label"] == 1).sum())

if len(df) < MIN_EXPECTED_ROWS:
    print()
    print("WARNING: The dataset only has", len(df), "rows.")
    print("The real Kaggle dataset has around 18,600 rows.")
    print("You may be using a small test file by mistake.")
    print("Training will continue, but the results will not be meaningful.")
    print()


print("\nStep 2: Cleaning the text...")

def clean_text(text):
    text = text.lower()
    # numtoken just replaces every number so the model doesn't care about specific amounts
    text = re.sub(r"http\S+", " urltoken ", text)
    text = re.sub(r"www\.\S+", " urltoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    for ch in string.punctuation:
        text = text.replace(ch, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_clean"] = df["text"].apply(clean_text)
df = df[df["text_clean"].str.len() > 0]

print("  example BEFORE:", df["text"].iloc[0][:80], "...")
print("  example AFTER :", df["text_clean"].iloc[0][:80], "...")


print("\nStep 3: Splitting into train and test sets...")

X = df["text_clean"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_SEED,
)

print("  training emails:", len(X_train))
print("  testing emails :", len(X_test))


print("\nStep 4: Building TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    stop_words="english",
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("  training feature matrix shape:", X_train_tfidf.shape)


print("\nStep 5: Training the models...")

models = {
    "SVM":           LinearSVC(random_state=RANDOM_SEED, max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    "XGBoost":       XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, eval_metric="logloss"),
}

for model_name in models:
    print("  training", model_name, "...")
    models[model_name].fit(X_train_tfidf, y_train)


print("\nStep 6: Evaluating the models...")

results = []
predictions = {}

for model_name in models:
    model = models[model_name]
    y_pred = model.predict(X_test_tfidf)
    predictions[model_name] = y_pred

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results.append({
        "model":     model_name,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    })

    print("\n  ===", model_name, "===")
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))

results_df = pd.DataFrame(results)


print("\nStep 7: Saving models and metrics...")

joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
for model_name in models:
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(models[model_name], os.path.join(models_dir, filename))

results_df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)


print("Step 8: Making the comparison chart...")

metric_names = ["accuracy", "precision", "recall", "f1"]
model_names = results_df["model"].tolist()
x_positions = np.arange(len(model_names))
bar_width = 0.2
colors = ["#2F3C7E", "#F96167", "#028090", "#B85042"]

plt.figure(figsize=(9, 5))
for i, metric in enumerate(metric_names):
    plt.bar(
        x_positions + i * bar_width,
        results_df[metric],
        bar_width,
        label=metric.capitalize(),
        color=colors[i],
    )
plt.xticks(x_positions + bar_width * 1.5, model_names)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Comparison - Phishing Email Classification")
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=150)
plt.close()


print("Step 9: Making confusion matrices...")

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))

for ax, model_name in zip(axes, models):
    y_pred = predictions[model_name]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["legit", "phishing"],
        yticklabels=["legit", "phishing"],
        cbar=False,
    )
    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrices.png"), dpi=150)
plt.close()


print("\n=============================================")
print("FINAL RESULTS")
print("=============================================")
print(results_df.to_string(index=False))
print()
print("Models saved in :", models_dir + "/")
print("Results saved in:", results_dir + "/")
print()
print("Next step: run")
print("  python src/3_classify_your_email.py")
print("to try the trained model on your own emails.")
