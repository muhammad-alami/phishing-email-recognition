"""
Train an XGBoost classifier on TF-IDF features from the processed email dataset.

This script:
  1. Loads data/processed/emails_clean.csv
  2. Applies the same 80/20 stratified split used by all three train scripts
  3. Loads the pre-fitted TF-IDF vectorizer
  4. Trains an XGBClassifier with 300 boosting rounds
  5. Saves the model to models/xgboost.joblib
  6. Saves test predictions and class probabilities to results/

XGBoost is the slowest of the three — 300 rounds on a 10,000-feature
sparse matrix takes about 5 minutes on my laptop. The eval_metric flag
just silences the default deprecation warning; I'm not using early stopping.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import build_or_load_vectorizer

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 300


def train(X_train, y_train) -> XGBClassifier:
    """Fit XGBClassifier with 300 boosting rounds."""
    clf = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    return clf


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "processed" / "emails_clean.csv"
    models_dir = root / "models"
    results_dir = root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_xgb] Loading {data_path.name}...")
    df = pd.read_csv(data_path)
    X = df["text_clean"].values
    y = df["label"].values
    print(f"[train_xgb] {len(df):,} rows, {y.mean():.1%} phishing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    vec = build_or_load_vectorizer(X_train, root)
    X_train_v = vec.transform(X_train)
    X_test_v = vec.transform(X_test)

    print(f"[train_xgb] Training XGBoost ({N_ESTIMATORS} rounds)...")
    clf = train(X_train_v, y_train)

    joblib.dump(clf, models_dir / "xgboost.joblib")
    print("[train_xgb] Saved model to models/xgboost.joblib")

    preds = clf.predict(X_test_v)
    scores = clf.predict_proba(X_test_v)[:, 1]

    np.save(results_dir / "preds_xgb.npy", preds)
    np.save(results_dir / "scores_xgb.npy", scores)
    np.save(results_dir / "y_test.npy", y_test)

    acc = (preds == y_test).mean()
    print(f"[train_xgb] Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
