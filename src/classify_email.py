"""
Interactive CLI for classifying a pasted email as phishing or legitimate.

This script:
  1. Loads the saved SVM model and TF-IDF vectorizer from models/
  2. Prompts the user to paste email text (type END on its own line to submit)
  3. Cleans and vectorizes the input using the same pipeline as training
  4. Prints the prediction and the raw decision function score
  5. Loops until the user types QUIT

I only use the SVM here because it's the fastest and most accurate of the
three. A decision score above 0 means phishing; larger positive values mean
higher confidence. I don't have access to email headers so this is the best
I can do with body text alone.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from clean_text import clean_text


def read_email() -> str:
    """Collect multi-line email input, terminated by END on its own line."""
    print("Paste your email text below. Type END on a new line when done:")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def classify(text: str, clf, vec) -> tuple[str, float]:
    """Clean, vectorize, and classify a single email. Returns (label, score)."""
    features = vec.transform([clean_text(text)])
    pred  = clf.predict(features)[0]
    score = float(clf.decision_function(features)[0])
    return ("PHISHING" if pred == 1 else "LEGITIMATE"), score


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"

    print("[classify_email] Loading model and vectorizer...")
    clf = joblib.load(models_dir / "svm.joblib")
    vec = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    print("[classify_email] Ready. Type QUIT to exit.\n")

    while True:
        try:
            text = read_email()
        except (EOFError, KeyboardInterrupt):
            break

        if text.strip().upper() == "QUIT":
            break

        if not text.strip():
            print("[classify_email] (empty input, try again)\n")
            continue

        label, score = classify(text, clf, vec)
        print(f"\n[classify_email] Result: {label}  (decision score: {score:+.4f})\n")

    print("[classify_email] Exiting.")


if __name__ == "__main__":
    main()
