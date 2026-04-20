# quick sanity check - runs the saved SVM on 3 example emails

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

print("Loading the saved SVM model...")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
model = joblib.load("models/svm.joblib")


example_emails = [
    # phishing
    """Dear Customer,
    We detected UNUSUAL activity on your account. Your account will be
    suspended in 24 hours unless you VERIFY your identity immediately.
    Click here: http://secure-paypa1-login.com/verify
    PayPal Security Team""",

    # legit
    """Hi team,
    Just a reminder that tomorrow's sprint planning meeting is moved
    to 2pm in conference room B. Please review the backlog before
    the meeting.
    Thanks, Sarah""",

    # phishing
    """CONGRATULATIONS! You have won a $1000 Amazon gift card!
    Click the link below to claim your prize before it expires:
    http://amzn-giftcards-free.ru/claim?id=9921
    Limited time offer!""",
]


print("\nClassifying", len(example_emails), "example emails...\n")

for i, email in enumerate(example_emails, start=1):
    cleaned = clean_text(email)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    snippet = email.strip().replace("\n", " ")[:70]

    print("Email #" + str(i))
    print("  snippet:", snippet, "...")
    print("  result :", label)
    print()
