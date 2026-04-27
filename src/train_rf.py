"""
Train a Random Forest on TF-IDF features from the processed email dataset.

This script:
  1. Loads data/processed/emails_clean.csv
  2. Applies the same 80/20 stratified split used by all three train scripts
  3. Loads the pre-fitted TF-IDF vectorizer (run train_svm.py first, or any
     train script — they all save to the same path)
  4. Trains a RandomForestClassifier with 200 trees
  5. Saves the model to models/random_forest.joblib
  6. Saves test predictions and class probabilities to results/

200 trees is a lot but the forest plateaus around 150 on this dataset —
I checked during the ablation. Expect about 2 minutes on a modern laptop.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features import build_or_load_vectorizer

TEST_SIZE = 0.2
RANDOM_STATE = 42

# 200 trees — same as the configuration in Fares et al. (2024) Table 3.
N_ESTIMATORS = 200


def train(X_train, y_train) -> RandomForestClassifier:
    """Fit a Random Forest with 200 trees."""
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "processed" / "emails_clean.csv"
    models_dir = root / "models"
    results_dir = root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_rf] Loading {data_path.name}...")
    df = pd.read_csv(data_path)
    X = df["text_clean"].values
    y = df["label"].values
    print(f"[train_rf] {len(df):,} rows, {y.mean():.1%} phishing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    vec = build_or_load_vectorizer(X_train, root)
    X_train_v = vec.transform(X_train)
    X_test_v = vec.transform(X_test)

    print(f"[train_rf] Training RandomForest ({N_ESTIMATORS} trees)...")
    clf = train(X_train_v, y_train)

    joblib.dump(clf, models_dir / "random_forest.joblib")
    print("[train_rf] Saved model to models/random_forest.joblib")

    preds = clf.predict(X_test_v)
    scores = clf.predict_proba(X_test_v)[:, 1]

    np.save(results_dir / "preds_rf.npy", preds)
    np.save(results_dir / "scores_rf.npy", scores)
    np.save(results_dir / "y_test.npy", y_test)

    acc = (preds == y_test).mean()
    print(f"[train_rf] Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
