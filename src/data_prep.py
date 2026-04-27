"""
Load and clean the raw phishing email CSV, then write a tidy processed file.

This script:
  1. Reads data/Phishing_Email.csv downloaded from Kaggle
  2. Renames the two relevant columns and maps label strings to 0/1 integers
  3. Applies clean_text() to every row
  4. Drops rows whose text is empty after cleaning (rare but happens with
     pure-HTML emails that strip down to nothing)
  5. Writes data/processed/emails_clean.csv with columns:
     id, text_raw, text_clean, label, length

Every downstream script reads from the processed file so the cleaning
logic only lives here. Running this takes about 30 seconds on my laptop.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from clean_text import clean_text

# Kaggle dataset uses these exact strings for the two classes. I lowercase
# before comparing just in case the formatting drifts across dataset versions.
LABEL_PHISHING = "phishing email"

# Raw CSV column names — these match the Kaggle download exactly.
COL_TEXT = "Email Text"
COL_TYPE = "Email Type"


def parse_label(raw: str) -> int:
    """Map the Kaggle label string to a 0/1 integer.

    Returns -1 for nulls so the caller can filter them out cleanly.
    """
    if pd.isna(raw):
        return -1  # caller drops these
    return 1 if str(raw).strip().lower() == LABEL_PHISHING else 0


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV and rename columns to something sane."""
    df = pd.read_csv(csv_path)
    # Kaggle sometimes adds an unnamed index column — drop it if present.
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df = df.rename(columns={COL_TEXT: "text_raw", COL_TYPE: "label_str"})
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text, assign integer labels, drop unusable rows.

    Returns a tidy dataframe with id, text_raw, text_clean, label, length.
    """
    df = df.copy()
    df["label"] = df["label_str"].apply(parse_label)

    # Drop rows where the label couldn't be parsed or the raw text is missing.
    df = df[df["label"] >= 0].dropna(subset=["text_raw"])

    print(f"[data_prep] Cleaning {len(df):,} rows...")
    df["text_clean"] = df["text_raw"].apply(clean_text)

    # A handful of rows become empty after cleaning (e.g. pure HTML with
    # no readable words). I don't have a way to recover them so I drop them.
    df = df[df["text_clean"].str.strip().str.len() > 0]

    df["length"] = df["text_raw"].str.len()
    df = df.reset_index(drop=True)
    df["id"] = df.index

    return df[["id", "text_raw", "text_clean", "label", "length"]]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "data" / "Phishing_Email.csv"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emails_clean.csv"

    print(f"[data_prep] Reading {csv_path.name}...")
    raw = load_raw(csv_path)
    print(f"[data_prep] Raw rows: {len(raw):,}")

    df = prepare(raw)
    phishing_rate = df["label"].mean()
    print(f"[data_prep] After cleaning: {len(df):,} rows, {phishing_rate:.1%} phishing")

    df.to_csv(out_path, index=False)
    print(f"[data_prep] Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
