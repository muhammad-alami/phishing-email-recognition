# paste an email, get back phishing or legit - uses the trained SVM

import os
import re
import string
import sys
import joblib


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " urltoken ", text)
    text = re.sub(r"www\.\S+", " urltoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    for ch in string.punctuation:
        text = text.replace(ch, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


if not os.path.exists("models/svm.joblib"):
    print("model not found, run 1_train_models.py first")
    sys.exit(1)

vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
model = joblib.load("models/svm.joblib")
print("model loaded\n")


def read_email_from_user():
    print("paste an email, type END when done, QUIT to exit")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "QUIT":
            return None
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


while True:
    email_text = read_email_from_user()

    if email_text is None:
        print("bye")
        break

    if len(email_text.strip()) == 0:
        print("(empty, try again)\n")
        continue

    cleaned = clean_text(email_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    score = model.decision_function(features)[0]

    label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    print(f"\n>>> {label} (score: {round(score, 2)})\n")
