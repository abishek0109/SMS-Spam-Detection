import re
import string
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


BASE_DIR = Path(__file__).resolve().parent
MAX_MESSAGE_LENGTH = 5000

app = Flask(__name__)

stemmer = PorterStemmer()
punctuation_translation = str.maketrans("", "", string.punctuation)

FALLBACK_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    STOP_WORDS = FALLBACK_STOP_WORDS


def load_artifacts():
    """Load each fitted estimator once when the web process starts."""
    artifacts = {
        "logistic": {
            "model": joblib.load(BASE_DIR / "lr_model.pkl"),
            "vectorizer": joblib.load(BASE_DIR / "tfidf_vectorizer.pkl"),
            "label": "Logistic Regression",
            "feature_label": "TF-IDF with bigrams",
        },
        "naive_bayes": {
            "model": joblib.load(BASE_DIR / "nb_model.pkl"),
            "vectorizer": joblib.load(BASE_DIR / "count_vectorizer.pkl"),
            "label": "Multinomial Naive Bayes",
            "feature_label": "Token counts",
        },
    }
    return artifacts


MODELS = load_artifacts()


def preprocess(text: str) -> str:
    """Apply the same normalization used while training the saved models."""
    normalized_text = str(text).lower()
    normalized_text = normalized_text.translate(punctuation_translation)
    normalized_text = re.sub(r"\d+", " ", normalized_text)
    normalized_text = re.sub(r"[^a-z\s]", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    tokens = normalized_text.split()
    cleaned_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(cleaned_tokens)


def classify_message(message: str, model_key: str) -> dict:
    """Return a user-facing prediction and useful model diagnostics."""
    artifact = MODELS[model_key]
    cleaned_message = preprocess(message)
    features = artifact["vectorizer"].transform([cleaned_message])
    spam_probability = float(artifact["model"].predict_proba(features)[0][1])
    predicted_class = int(spam_probability >= 0.5)
    confidence = spam_probability if predicted_class else 1.0 - spam_probability

    return {
        "prediction": "spam" if predicted_class else "ham",
        "confidence": round(confidence, 4),
        "spam_probability": round(spam_probability, 4),
        "ham_probability": round(1.0 - spam_probability, 4),
        "model": artifact["label"],
        "features": artifact["feature_label"],
        "processed_text": cleaned_message,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "models": list(MODELS)})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    model_key = payload.get("model", "logistic")

    if not message:
        return jsonify({"error": "Enter an SMS message to analyze."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify(
            {"error": f"Message must be {MAX_MESSAGE_LENGTH} characters or less."}
        ), 400
    if model_key not in MODELS:
        return jsonify({"error": "The selected model is not available."}), 400

    result = classify_message(message, model_key)
    return jsonify({"message": message, **result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
