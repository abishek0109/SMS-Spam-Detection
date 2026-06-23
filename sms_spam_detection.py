# ==================================================
# SECTION 1: IMPORTS
# ==================================================
import os
import re
import string
import warnings

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

try:
    nltk.download("punkt")
    nltk.download("stopwords")
except Exception as nltk_download_error:
    warnings.warn(
        "NLTK resource download failed. The script will continue with "
        "offline-safe preprocessing fallbacks if local NLTK data is missing. "
        f"Original error: {nltk_download_error}",
        UserWarning,
    )


# ==================================================
# SECTION 2: DATA LOADING & EDA
# ==================================================
RANDOM_STATE = 42
DATASET_PATH = "spam.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

sns.set_theme(style="whitegrid", palette="deep")
np.random.seed(RANDOM_STATE)


def load_sms_dataset(file_path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the SMS Spam Collection with a portable fallback.

    The UCI CSV is commonly stored locally as spam.csv with latin-1 encoding.
    If the file is absent, the script attempts to load an equivalent raw
    GitHub copy so notebooks and Colab sessions can still run end-to-end.
    """
    try:
        dataframe = pd.read_csv(file_path, encoding="latin-1")
        print(f"Loaded local dataset from: {os.path.abspath(file_path)}")
        return dataframe
    except FileNotFoundError:
        warnings.warn(
            f"Local file '{file_path}' was not found. Attempting to load a "
            "raw GitHub copy of the SMS Spam Collection instead.",
            UserWarning,
        )
        try:
            dataframe = pd.read_csv(
                DATASET_URL,
                sep="\t",
                names=["v1", "v2"],
                encoding="latin-1",
            )
            print(f"Loaded fallback dataset from: {DATASET_URL}")
            return dataframe
        except Exception as error:
            raise FileNotFoundError(
                "Could not load the dataset locally or from the fallback URL. "
                "Place spam.csv in the working directory and rerun the script."
            ) from error


df = load_sms_dataset()

# The UCI CSV often contains extra unnamed columns caused by delimiter artifacts.
# They do not carry predictive signal, so dropping them avoids noisy null-heavy
# fields leaking into later analysis.
unnamed_columns = [column for column in df.columns if column.startswith("Unnamed")]
df = df.drop(columns=unnamed_columns)
df = df.rename(columns={"v1": "label", "v2": "message"})

# Keep only the two fields required for a binary text classification pipeline.
df = df[["label", "message"]]

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nNull counts:")
print(df.isnull().sum())

print("\nDuplicate count:")
print(df.duplicated().sum())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nFive random HAM messages:")
print(
    df.loc[df["label"] == "ham", "message"]
    .sample(5, random_state=RANDOM_STATE)
    .to_string(index=False)
)

print("\nFive random SPAM messages:")
print(
    df.loc[df["label"] == "spam", "message"]
    .sample(5, random_state=RANDOM_STATE)
    .to_string(index=False)
)

plt.figure(figsize=(8, 5))
class_counts = df["label"].value_counts().sort_index()
class_axis = sns.barplot(
    x=class_counts.index,
    y=class_counts.values,
    hue=class_counts.index,
    palette={"ham": "#4C78A8", "spam": "#F58518"},
    legend=False,
)
class_axis.set_title("SMS Class Distribution")
class_axis.set_xlabel("Class")
class_axis.set_ylabel("Number of Messages")
for container in class_axis.containers:
    class_axis.bar_label(container, fmt="%d")
plt.tight_layout()
plt.show()

# Message length is not the primary model input here, but it is useful EDA:
# spam campaigns often include links, prize text, or call-to-action language
# that can make messages longer than conversational ham.
df["message_length"] = df["message"].astype(str).str.len()

plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="message_length",
    hue="label",
    bins=50,
    kde=True,
    stat="density",
    common_norm=False,
    palette={"ham": "#4C78A8", "spam": "#F58518"},
    alpha=0.45,
)
plt.title("Message Length Distribution by Class")
plt.xlabel("Message Length in Characters")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

ham_count = int(class_counts.get("ham", 0))
spam_count = int(class_counts.get("spam", 0))
spam_rate = spam_count / len(df)
ham_length_mean = df.loc[df["label"] == "ham", "message_length"].mean()
spam_length_mean = df.loc[df["label"] == "spam", "message_length"].mean()

print(
    "\nEDA commentary: The dataset is imbalanced because ham messages are much "
    f"more common than spam messages ({ham_count} ham vs {spam_count} spam; "
    f"spam rate = {spam_rate:.2%}). Spam messages also tend to be longer on "
    f"average in this sample ({spam_length_mean:.1f} characters for spam vs "
    f"{ham_length_mean:.1f} for ham), which is a useful behavioral clue even "
    "though the main model features are token based."
)


# ==================================================
# SECTION 3: TEXT PREPROCESSING
# ==================================================
stemmer = PorterStemmer()
punctuation_translation = str.maketrans("", "", string.punctuation)

FALLBACK_STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    warnings.warn(
        "NLTK stopwords corpus was not found. Using a built-in English "
        "stopword fallback so the script can run in offline environments.",
        UserWarning,
    )
    stop_words = FALLBACK_STOP_WORDS


def preprocess(text):
    """Normalize SMS text into a compact, model-friendly token string."""
    text = str(text).lower()

    # Punctuation, digits, and special characters inflate vocabulary size.
    # Removing them makes traditional bag-of-words features less sparse.
    text = text.translate(punctuation_translation)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # word_tokenize depends on the punkt resource. wordpunct_tokenize is
        # part of NLTK itself, so it keeps offline runs functional while the
        # primary path still uses the requested tokenizer when data exists.
        tokens = nltk.wordpunct_tokenize(text)

    # Stopwords are removed because common function words usually add little
    # signal for spam detection while increasing feature dimensionality.
    cleaned_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return " ".join(cleaned_tokens)


print("\nPreprocessing examples:")
example_messages = df["message"].sample(5, random_state=RANDOM_STATE)
for index, original_message in example_messages.items():
    print(f"\nExample index: {index}")
    print(f"Before: {original_message}")
    print(f"After:  {preprocess(original_message)}")


# ==================================================
# SECTION 4: FEATURE ENGINEERING
# ==================================================
df["clean_message"] = df["message"].apply(preprocess)
df["target"] = df["label"].map({"ham": 0, "spam": 1})

if df["target"].isnull().any():
    raise ValueError("Unexpected labels found. Expected only 'ham' and 'spam'.")

X = df["clean_message"]
y = df["target"].astype(int)

# CountVectorizer is a natural match for MultinomialNB because the model was
# designed around count-like event frequencies, making raw token occurrence
# features a strong and interpretable baseline for text classification.
count_vectorizer = CountVectorizer(max_features=5000)

# Logistic Regression benefits from TF-IDF because weighted features reduce the
# dominance of frequent generic terms and make discriminative tokens and short
# phrases more linearly separable. Bigrams capture signals such as "free entry"
# or "call now" that unigrams may split apart.
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))


# ==================================================
# SECTION 5: TRAIN TEST SPLIT
# ==================================================
# stratify=y preserves the spam/ham ratio in both splits. This is critical for
# imbalanced datasets because an unlucky split could underrepresent spam in the
# test set and produce misleadingly optimistic or unstable metrics.
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain/Test split sizes:")
print(f"Training samples: {len(X_train_text)}")
print(f"Testing samples:  {len(X_test_text)}")


def print_split_ratio(split_name: str, labels: pd.Series) -> None:
    """Print ham/spam counts and percentages for a target split."""
    counts = labels.value_counts().sort_index()
    ham = int(counts.get(0, 0))
    spam = int(counts.get(1, 0))
    total = len(labels)
    print(
        f"{split_name}: ham={ham} ({ham / total:.2%}), "
        f"spam={spam} ({spam / total:.2%})"
    )


print("\nSpam/Ham ratio by split:")
print_split_ratio("Train", y_train)
print_split_ratio("Test ", y_test)

X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)


# ==================================================
# SECTION 6: MODEL TRAINING
# ==================================================
nb_model = MultinomialNB()
nb_model.fit(X_train_count, y_train)

lr_model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    C=1.0,
    random_state=RANDOM_STATE,
)
lr_model.fit(X_train_tfidf, y_train)


# ==================================================
# SECTION 7: MODEL EVALUATION
# ==================================================
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a probabilistic binary classifier with metrics and plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\nClassification Report: {model_name}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} (AUC = {roc_auc:.4f})",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
    }


nb_metrics = evaluate_model(
    nb_model,
    X_test_count,
    y_test,
    "Multinomial Naive Bayes + CountVectorizer",
)
lr_metrics = evaluate_model(
    lr_model,
    X_test_tfidf,
    y_test,
    "Logistic Regression + TF-IDF",
)


# ==================================================
# SECTION 8: MODEL COMPARISON
# ==================================================
comparison_df = pd.DataFrame([nb_metrics, lr_metrics])
comparison_df = comparison_df.sort_values(by="ROC-AUC", ascending=False)

print("\nModel comparison:")
print(comparison_df.to_string(index=False, float_format="{:.4f}".format))

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
plot_df = comparison_df.melt(
    id_vars="Model",
    value_vars=metrics_to_plot,
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=plot_df,
    x="Metric",
    y="Score",
    hue="Model",
    palette=["#4C78A8", "#F58518"],
)
plt.title("Model Metric Comparison")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend(title="Model", loc="lower right")
plt.tight_layout()
plt.show()

best_model_row = comparison_df.iloc[0]
print(
    "\nModel interpretation: "
    f"{best_model_row['Model']} ranks highest by ROC-AUC "
    f"({best_model_row['ROC-AUC']:.4f}). ROC-AUC is emphasized because it "
    "measures ranking quality across decision thresholds, which is helpful "
    "for imbalanced spam detection. Precision and recall should still be "
    "reviewed together because production systems often need to balance "
    "blocking spam against avoiding false alarms on legitimate messages."
)


# ==================================================
# SECTION 9: INFERENCE PIPELINE
# ==================================================
def predict_sms(message: str, model, vectorizer):
    """Preprocess, vectorize, and classify a single SMS message."""
    cleaned_message = preprocess(message)
    message_features = vectorizer.transform([cleaned_message])
    predicted_class = int(model.predict(message_features)[0])
    predicted_probability = float(model.predict_proba(message_features)[0][1])

    if predicted_class == 1:
        prediction = "spam"
        confidence = predicted_probability
    else:
        prediction = "ham"
        confidence = 1.0 - predicted_probability

    return {
        "message": message,
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }


test_messages = [
    "Congratulations! You have won a free iPhone. Click now!",
    "Hey, are we still meeting at 6 PM today?",
    "Please call me when you get a chance.",
]

print("\nInference examples using Logistic Regression + TF-IDF:")
for sms_message in test_messages:
    print(predict_sms(sms_message, lr_model, tfidf_vectorizer))


# ==================================================
# SECTION 10: SAVE ARTIFACTS
# ==================================================
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(count_vectorizer, "count_vectorizer.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print("\nSaved artifacts:")
print("nb_model.pkl")
print("lr_model.pkl")
print("count_vectorizer.pkl")
print("tfidf_vectorizer.pkl")

# Demonstration snippet: reload a saved model into a fresh variable and use it
# exactly as any fitted scikit-learn estimator.
reloaded_lr_model = joblib.load("lr_model.pkl")
print("\nReloaded model demonstration:")
print(predict_sms("Win cash now by replying YES.", reloaded_lr_model, tfidf_vectorizer))
