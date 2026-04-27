"""
Text normalization for phishing email detection.

This module:
  1. Defines the replacement tokens and punctuation list shared across the project
  2. Exports clean_text(), the single function all other modules call

I keep this in its own file so data_prep.py and classify_email.py import
the exact same normalization — one definition, no drift between scripts.
"""
from __future__ import annotations

import re
import string

# Replacement tokens — descriptive words so the model can treat "urltoken"
# and "numtoken" as real informative features rather than just noise.
URL_TOKEN = "urltoken"
NUM_TOKEN = "numtoken"


def clean_text(text: str) -> str:
    """Normalize email text so the model sees consistent input.

    Lowercase, replace URLs with urltoken, replace numbers with numtoken,
    strip punctuation, collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", f" {URL_TOKEN} ", text)
    text = re.sub(r"www\.\S+", f" {URL_TOKEN} ", text)
    text = re.sub(r"\d+", f" {NUM_TOKEN} ", text)
    for ch in string.punctuation:
        text = text.replace(ch, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text
