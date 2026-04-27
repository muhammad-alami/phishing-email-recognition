"""
Train a LinearSVC on TF-IDF features from the processed email dataset.

This script:
  1. Loads data/processed/emails_clean.csv
  2. Splits 80/20 stratified by label (random_state=42)
  3. Builds or loads the shared TF-IDF vectorizer (see features.py)
  4. Trains a LinearSVC with max_iter=2000
  5. Saves the model to models/svm.joblib
  6. Saves test predictions and decision scores to results/ for evaluate.py

Fares et al. (2024) report SVM as their strongest baseline on TF-IDF
features, which is why I'm starting here. This runs in about 15 seconds
on my laptop once the vectorizer is already built.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from features import build_or_load_vectorizer

# I fix the seed so evaluate.py sees the same test set regardless of
# which train script happened to run last.
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train(X_train, y_train) -> LinearSVC:
    """Fit LinearSVC with the hyperparameters from the paper."""
    clf = LinearSVC(random_state=RANDOM_STATE, max_iter=2_000)
    clf.fit(X_train, y_train)
    return clf


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "processed" / "emails_clean.csv"
    models_dir = root / "models"
    results_dir = root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_svm] Loading {data_path.name}...")
    df = pd.read_csv(data_path)
    X = df["text_clean"].values
    y = df["label"].values
    print(f"[train_svm] {len(df):,} rows, {y.mean():.1%} phishing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    vec = build_or_load_vectorizer(X_train, root)
    X_train_v = vec.transform(X_train)
    X_test_v = vec.transform(X_test)

    print("[train_svm] Training LinearSVC...")
    clf = train(X_train_v, y_train)

    joblib.dump(clf, models_dir / "svm.joblib")
    print("[train_svm] Saved model to models/svm.joblib")

    preds = clf.predict(X_test_v)
    scores = clf.decision_function(X_test_v)

    np.save(results_dir / "preds_svm.npy", preds)
    np.save(results_dir / "scores_svm.npy", scores)
    # y_test is the same for all three train scripts (same split), so any
    # of them can write it — evaluate.py and make_figures.py just need one copy.
    np.save(results_dir / "y_test.npy", y_test)

    acc = (preds == y_test).mean()
    print(f"[train_svm] Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
