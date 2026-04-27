"""
Generate all evaluation figures for the phishing email classification project.

This script:
  1. Loads saved predictions, scores, and models from results/ and models/
  2. Plots confusion matrices for all three classifiers
  3. Plots a grouped bar chart comparing all four metrics across models
  4. Plots ROC curves and Precision-Recall curves on shared axes
  5. Fits a calibrated SVM and plots a reliability diagram for all three models
  6. Plots a horizontal bar chart of the top-20 SVM feature coefficients
  7. Runs a 10-seed stability study and saves results/metrics_averaged.csv
  8. Runs the ablation study (unigrams-only and no-preprocessing variants)
  9. Runs error analysis on SVM misclassifications
 10. Draws the end-to-end pipeline architecture diagram

Run this after evaluate.py. The 10-seed study retrains all three models
from scratch so it takes about 15 minutes on my laptop.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from features import MAX_FEATURES, MIN_DF, NGRAM_RANGE

warnings.filterwarnings("ignore")

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Consistent colors across all figures — I only care about legibility,
# not matching a particular palette.
C_SVM = "#1f77b4"
C_RF  = "#ff7f0e"
C_XGB = "#2ca02c"


def load_artifacts(root: Path) -> dict:
    """Load all saved models, predictions, scores, and the processed dataset."""
    res = root / "results"
    mdl = root / "models"

    d = {
        "y_test":     np.load(res / "y_test.npy"),
        "preds_svm":  np.load(res / "preds_svm.npy"),
        "preds_rf":   np.load(res / "preds_rf.npy"),
        "preds_xgb":  np.load(res / "preds_xgb.npy"),
        "scores_svm": np.load(res / "scores_svm.npy"),
        "scores_rf":  np.load(res / "scores_rf.npy"),
        "scores_xgb": np.load(res / "scores_xgb.npy"),
        "svm": joblib.load(mdl / "svm.joblib"),
        "rf":  joblib.load(mdl / "random_forest.joblib"),
        "xgb": joblib.load(mdl / "xgboost.joblib"),
        "vec": joblib.load(mdl / "tfidf_vectorizer.joblib"),
        "df":  pd.read_csv(root / "data" / "processed" / "emails_clean.csv"),
    }
    return d


def _make_split(df: pd.DataFrame):
    """Reproduce the exact same train/test split used during training."""
    X = df["text_clean"].values
    y = df["label"].values
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def plot_confusion_matrices(d: dict, out_dir: Path) -> None:
    """Side-by-side confusion matrices for all three models."""
    specs = [
        ("SVM",           d["preds_svm"]),
        ("Random Forest", d["preds_rf"]),
        ("XGBoost",       d["preds_xgb"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, preds) in zip(axes, specs):
        cm = confusion_matrix(d["y_test"], preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phishing"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=13, fontweight="bold")
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[make_figures] Saved {path.name}")


def plot_model_comparison(d: dict, out_dir: Path) -> None:
    """Grouped bar chart comparing all four metrics across the three models."""
    y_test = d["y_test"]
    model_names = ["SVM", "Random Forest", "XGBoost"]
    preds_list  = [d["preds_svm"], d["preds_rf"], d["preds_xgb"]]
    metrics = {
        "Accuracy":  [accuracy_score(y_test, p) for p in preds_list],
        "Precision": [precision_score(y_test, p, zero_division=0) for p in preds_list],
        "Recall":    [recall_score(y_test, p, zero_division=0) for p in preds_list],
        "F1-Score":  [f1_score(y_test, p, zero_division=0) for p in preds_list],
    }

    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#2F3C7E", "#F96167", "#028090", "#B85042"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric_name, color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Phishing Email Classification", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[make_figures] Saved {path.name}")


def plot_roc_curves(d: dict, out_dir: Path) -> None:
    """ROC curves for all three models on the same axes."""
    y_test = d["y_test"]
    specs = [
        ("SVM",           d["scores_svm"], C_SVM),
        ("Random Forest", d["scores_rf"],  C_RF),
        ("XGBoost",       d["scores_xgb"], C_XGB),
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores, color in specs:
        auc = roc_auc_score(y_test, scores)
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random  (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    path = out_dir / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[make_figures] Saved {path.name}")
    auc_str = ", ".join(f"{n}: {roc_auc_score(y_test, s):.4f}" for n, s, _ in specs)
    print(f"[make_figures] AUC — {auc_str}")


def plot_pr_curves(d: dict, out_dir: Path) -> None:
    """Precision-Recall curves for all three models on the same axes."""
    y_test = d["y_test"]
    specs = [
        ("SVM",           d["scores_svm"], C_SVM),
        ("Random Forest", d["scores_rf"],  C_RF),
        ("XGBoost",       d["scores_xgb"], C_XGB),
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores, color in specs:
        ap = average_precision_score(y_test, scores)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{name}  (AP = {ap:.4f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    path = out_dir / "pr_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[make_figures] Saved {path.name}")
    ap_str = ", ".join(f"{n}: {average_precision_score(y_test, s):.4f}" for n, s, _ in specs)
    print(f"[make_figures] AP  — {ap_str}")


def plot_calibration(d: dict, X_train_v, y_train, X_test_v, out_dir: Path) -> None:
    """Reliability diagram: predicted probability vs. actual phishing rate.

    LinearSVC doesn't output probabilities natively, so I wrap it with
    CalibratedClassifierCV (sigmoid method, same as Platt scaling).
    """
    print("[make_figures] Fitting calibrated SVM (CalibratedClassifierCV)...")
    cal_svm = CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, max_iter=2_000), cv=5, method="sigmoid"
    )
    cal_svm.fit(X_train_v, y_train)
    cal_proba = cal_svm.predict_proba(X_test_v)[:, 1]

    y_test = d["y_test"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for proba, name, color in [
        (cal_proba,       "SVM (calibrated)", C_SVM),
        (d["scores_rf"],  "Random Forest",    C_RF),
        (d["scores_xgb"], "XGBoost",          C_XGB),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "s-", color=color, linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives",       fontsize=12)
    ax.set_title("Calibration / Reliability Diagram", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    path = out_dir / "calibration_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[make_figures] Saved {path.name}")


def plot_feature_importance(d: dict, out_dir: Path) -> None:
    """Horizontal bar chart: top-20 phishing and legit features by SVM coefficient."""
    names = d["vec"].get_feature_names_out()
    coef  = d["svm"].coef_.ravel()

    top_phi_idx = np.argsort(coef)[-20:][::-1]
    top_leg_idx = np.argsort(coef)[:20]
    phi_words, phi_scores = names[top_phi_idx], coef[top_phi_idx]
    leg_words, leg_scores = names[top_leg_idx], coef[top_leg_idx]

    y_pos = np.arange(20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    ax1.barh(y_pos, phi_scores[::-1], color="#d62728", edgecolor="black", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(phi_words[::-1], fontsize=9)
    ax1.set_title("Top 20 Phishing Indicators", fontsize=12, color="#d62728", fontweight="bold")
    ax1.set_xlabel("SVM Coefficient", fontsize=11)
    ax1.axvline(0, color="black", linewidth=0.8)

    ax2.barh(y_pos, leg_scores[::-1], color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(leg_words[::-1], fontsize=9)
    ax2.set_title("Top 20 Legitimate Indicators", fontsize=12, color="#2ca02c", fontweight="bold")
    ax2.set_xlabel("SVM Coefficient", fontsize=11)
    ax2.axvline(0, color="black", linewidth=0.8)

    fig.suptitle(
        "SVM Feature Importance — Top 20 Phishing vs Legitimate Indicators",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[make_figures] Saved {path.name}")


def plot_error_analysis(d: dict, out_dir: Path) -> None:
    """Bar chart of average email length for correctly classified, FN, and FP emails."""
    y_test = d["y_test"]
    preds  = d["preds_svm"]
    df     = d["df"]

    # I only need the raw text for the test portion — same split parameters
    # as the train scripts, so I get the same partition.
    _, orig_test, _, _ = train_test_split(
        df["text_raw"].values, df["label"].values,
        test_size=TEST_SIZE, stratify=df["label"].values, random_state=RANDOM_STATE,
    )

    fn_mask      = (y_test == 1) & (preds == 0)
    fp_mask      = (y_test == 0) & (preds == 1)
    correct_mask = y_test == preds

    def avg_len(texts):
        return float(np.mean([len(str(t)) for t in texts])) if len(texts) > 0 else 0.0

    len_correct = avg_len(orig_test[correct_mask])
    len_fn      = avg_len(orig_test[fn_mask])
    len_fp      = avg_len(orig_test[fp_mask])

    print(f"[make_figures] SVM errors — FN: {fn_mask.sum():,}, FP: {fp_mask.sum():,}")
    print(f"[make_figures] Avg length — correct: {len_correct:.0f}, FN: {len_fn:.0f}, FP: {len_fp:.0f}")

    cats    = ["Correctly\nClassified", "False Negatives\n(missed phishing)", "False Positives\n(false alarm)"]
    lengths = [len_correct, len_fn, len_fp]
    colors  = ["#2ca02c", "#d62728", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(cats, lengths, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, lengths):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(lengths) * 0.01,
            f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Average Character Length", fontsize=12)
    ax.set_title("Average Email Length by Classification Outcome (SVM)", fontsize=12)
    ax.set_ylim(0, max(lengths) * 1.15)
    fig.tight_layout()
    path = out_dir / "error_analysis_lengths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[make_figures] Saved {path.name}")


def plot_pipeline_diagram(out_dir: Path) -> None:
    """Block diagram showing the end-to-end classification pipeline."""
    BG     = "#1a0a2e"
    PURPLE = "#4a1a8c"
    CYAN   = "#0a6a7c"
    TEXT   = "#e8e8ff"
    ARROW  = "#00d4ff"

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    BOX_H = 1.6
    BOX_Y = 1.4
    boxes = [
        (0.2,  BOX_Y, 2.2, BOX_H, "Raw Email",                    PURPLE),
        (3.0,  BOX_Y, 2.2, BOX_H, "clean_text()",                 CYAN),
        (5.8,  BOX_Y, 2.4, BOX_H, "TF-IDF\nVectorizer",           PURPLE),
        (8.8,  BOX_Y, 2.6, BOX_H, "Classifier\n(SVM / RF / XGB)", CYAN),
        (12.0, BOX_Y, 1.8, BOX_H, "Prediction",                   PURPLE),
    ]
    mid_y = BOX_Y + BOX_H / 2

    for x, y_b, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y_b), w, h, boxstyle="round,pad=0.12",
            linewidth=2, edgecolor=ARROW, facecolor=color,
        ))
        ax.text(
            x + w / 2, y_b + h / 2, label,
            ha="center", va="center",
            color=TEXT, fontsize=10.5, fontweight="bold", multialignment="center",
        )

    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + boxes[i][2]
        x_end   = boxes[i + 1][0]
        ax.annotate(
            "", xy=(x_end, mid_y), xytext=(x_start, mid_y),
            arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=2.2, mutation_scale=20),
        )

    ax.set_title(
        "Phishing Email Classification — End-to-End Pipeline",
        color=TEXT, fontsize=14, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    path = out_dir / "pipeline_diagram.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[make_figures] Saved {path.name}")


def run_seed_study(df: pd.DataFrame, out_dir: Path) -> None:
    """Re-train all three models on 10 random splits and report mean +/- std.

    This is the same stability study from 4_make_extra_figures.py. It takes
    a while (each of the 10 seeds trains three models from scratch).
    """
    print("[make_figures] Starting 10-seed stability study...")
    X = df["text_clean"].values
    y = df["label"].values

    seed_results = {m: {"accuracy": [], "precision": [], "recall": [], "f1": []}
                    for m in ["SVM", "Random Forest", "XGBoost"]}

    for seed in range(10):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        vec_s = TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES,
                                min_df=MIN_DF, stop_words="english")
        Xtr_v = vec_s.fit_transform(Xtr)
        Xte_v = vec_s.transform(Xte)

        clfs = {
            "SVM":          LinearSVC(random_state=42, max_iter=2_000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "XGBoost":      XGBClassifier(n_estimators=300, random_state=42, eval_metric="logloss"),
        }
        for name, clf in clfs.items():
            clf.fit(Xtr_v, ytr)
            p = clf.predict(Xte_v)
            seed_results[name]["accuracy"].append(accuracy_score(yte, p))
            seed_results[name]["precision"].append(precision_score(yte, p, zero_division=0))
            seed_results[name]["recall"].append(recall_score(yte, p, zero_division=0))
            seed_results[name]["f1"].append(f1_score(yte, p, zero_division=0))
        print(f"[make_figures] Seed {seed} done.")

    rows = []
    for model_name, metrics in seed_results.items():
        row = {"Model": model_name}
        for metric, vals in metrics.items():
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 6)
            row[f"{metric}_std"]  = round(float(np.std(vals)),  6)
        rows.append(row)

    path = out_dir / "metrics_averaged.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[make_figures] Saved {path.name}")


def run_ablation_study(d: dict) -> None:
    """Train two SVM variants on the original split and print the comparison.

    Variant 1 — unigrams only (no bigrams).
    Variant 2 — raw lowercase text with no clean_text() preprocessing.
    I don't have a baseline to beat here, just checking how much each
    preprocessing step contributes to the final F1.
    """
    df     = d["df"]
    y_test = d["y_test"]

    # Reproduce the original split to get the train/test text arrays.
    X_clean = df["text_clean"].values
    y       = df["label"].values
    X_train_clean, X_test_clean, y_train, _ = train_test_split(
        X_clean, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Reference: full pipeline accuracy (from saved predictions).
    ref_acc = accuracy_score(y_test, d["preds_svm"])
    ref_f1  = f1_score(y_test, d["preds_svm"], zero_division=0)

    # Variant 1: unigrams only.
    vec_uni = TfidfVectorizer(ngram_range=(1, 1), max_features=MAX_FEATURES,
                              min_df=MIN_DF, stop_words="english")
    svm_uni = LinearSVC(random_state=RANDOM_STATE, max_iter=2_000)
    svm_uni.fit(vec_uni.fit_transform(X_train_clean), y_train)
    preds_uni = svm_uni.predict(vec_uni.transform(X_test_clean))
    acc_uni = accuracy_score(y_test, preds_uni)
    f1_uni  = f1_score(y_test, preds_uni, zero_division=0)

    # Variant 2: raw lowercase text — skip clean_text() entirely.
    X_raw = df["text_raw"].str.lower().values
    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    vec_raw = TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES,
                              min_df=MIN_DF, stop_words="english")
    svm_raw = LinearSVC(random_state=RANDOM_STATE, max_iter=2_000)
    svm_raw.fit(vec_raw.fit_transform(X_train_raw), y_train)
    preds_raw = svm_raw.predict(vec_raw.transform(X_test_raw))
    acc_raw = accuracy_score(y_test, preds_raw)
    f1_raw  = f1_score(y_test, preds_raw, zero_division=0)

    print("[make_figures] Ablation study results:")
    print(f"[make_figures]   {'Variant':<48} {'Acc':>7} {'F1':>7}")
    print(f"[make_figures]   {'-' * 64}")
    print(f"[make_figures]   {'Full pipeline (bigrams + preprocessing)':<48} {ref_acc:.4f}  {ref_f1:.4f}")
    print(f"[make_figures]   {'Unigrams only [ngram_range=(1,1)]':<48} {acc_uni:.4f}  {f1_uni:.4f}")
    print(f"[make_figures]   {'No preprocessing (raw lowercase)':<48} {acc_raw:.4f}  {f1_raw:.4f}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[make_figures] Loading all saved artifacts...")
    d = load_artifacts(root)
    print(f"[make_figures] Test set: {len(d['y_test']):,} samples")

    # Build the vectorized train/test matrices once — calibration, ablation,
    # and error analysis all need them, so no point re-splitting each time.
    X_train_text, X_test_text, y_train, _ = _make_split(d["df"])
    X_train_v = d["vec"].transform(X_train_text)
    X_test_v  = d["vec"].transform(X_test_text)

    plot_confusion_matrices(d, out_dir)
    plot_model_comparison(d, out_dir)
    plot_roc_curves(d, out_dir)
    plot_pr_curves(d, out_dir)
    plot_calibration(d, X_train_v, y_train, X_test_v, out_dir)
    plot_feature_importance(d, out_dir)
    plot_error_analysis(d, out_dir)
    plot_pipeline_diagram(out_dir)
    run_ablation_study(d)
    run_seed_study(d["df"], out_dir)

    print("[make_figures] Done. All outputs in results/")


if __name__ == "__main__":
    main()
