"""
Load saved predictions and compute classification metrics for all three models.

This script:
  1. Reads results/y_test.npy and results/preds_*.npy written by the train scripts
  2. Computes accuracy, precision, recall, and F1 for SVM, Random Forest, XGBoost
  3. Writes results/metrics.csv (same format as before, for backwards compat)
  4. Writes results/metrics.json for any downstream script that wants structured data

Run this after all three train_*.py scripts have finished.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Short keys match the filenames (preds_svm.npy etc.). Display names go in the CSV.
MODELS = ["svm", "rf", "xgb"]
DISPLAY = {"svm": "SVM", "rf": "Random Forest", "xgb": "XGBoost"}


def load_predictions(results_dir: Path) -> dict:
    """Load y_test and all three sets of predictions from results/."""
    data = {"y_test": np.load(results_dir / "y_test.npy")}
    for m in MODELS:
        data[f"preds_{m}"] = np.load(results_dir / f"preds_{m}.npy")
    return data


def compute_metrics(y_true, y_pred) -> dict:
    """Return the four standard classification metrics as a dict."""
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


def main() -> None:
    import json

    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[evaluate] Loading saved predictions...")
    data = load_predictions(results_dir)
    y_test = data["y_test"]
    print(f"[evaluate] Test set: {len(y_test):,} samples, {y_test.mean():.1%} phishing")

    rows = []
    metrics_dict = {}
    for m in MODELS:
        preds = data[f"preds_{m}"]
        m_metrics = compute_metrics(y_test, preds)
        display = DISPLAY[m]
        rows.append({"model": display, **m_metrics})
        metrics_dict[display] = m_metrics
        print(
            f"[evaluate] {display:<15} "
            f"acc={m_metrics['accuracy']:.4f}  "
            f"prec={m_metrics['precision']:.4f}  "
            f"rec={m_metrics['recall']:.4f}  "
            f"f1={m_metrics['f1']:.4f}"
        )

    csv_path = results_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[evaluate] Wrote {csv_path.name}")

    json_path = results_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[evaluate] Wrote {json_path.name}")


if __name__ == "__main__":
    main()
