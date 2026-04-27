"""
TF-IDF vectorizer setup for phishing email detection.

This module:
  1. Defines the vectorizer hyperparameters as module-level constants
  2. Exports build_or_load_vectorizer(), which fits and saves the vectorizer
     on first call and just loads it on subsequent calls

Keeping the fit-or-load logic here means all three train scripts share the
exact same vocabulary without re-fitting on each run. The settings roughly
match those reported in Fares et al. (2024) for their SVM baseline — I
tested a few variants in the ablation and these gave the strongest F1.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF settings — chosen to match Fares et al. (2024). Bigrams help the
# model pick up two-word spam phrases like "click here" and "free offer".
MAX_FEATURES = 10_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
STOP_WORDS = "english"

# Path relative to project root — shared by all train scripts.
VECTORIZER_REL = "models/tfidf_vectorizer.joblib"


def make_vectorizer() -> TfidfVectorizer:
    """Return a fresh unfitted vectorizer with the project's standard settings."""
    return TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        stop_words=STOP_WORDS,
    )


def build_or_load_vectorizer(X_train, root: Path) -> TfidfVectorizer:
    """Fit and save the vectorizer, or just load it if it already exists.

    Skipping the fit when the file is present means train_rf.py and
    train_xgb.py use the identical vocabulary as train_svm.py without
    having to coordinate explicitly.
    """
    vec_path = root / VECTORIZER_REL
    if vec_path.exists():
        print(f"[features] Loading existing vectorizer from {vec_path.name}")
        return joblib.load(vec_path)

    print(f"[features] Fitting vectorizer on {len(X_train):,} training docs...")
    vec = make_vectorizer()
    vec.fit(X_train)
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, vec_path)
    print(f"[features] Saved vectorizer to {vec_path.name}")
    return vec
