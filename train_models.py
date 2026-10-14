from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Dataset" / "spam_edited_SMS.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 80


def ensure_nltk_resources() -> None:
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for resource_path, package in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: object) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def load_dataset(path: Path) -> tuple[pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the SMS CSV in the Dataset folder."
        )
    df = pd.read_csv(path)
    if {"v1", "v2"}.issubset(df.columns):
        labels, messages = df["v1"], df["v2"]
    elif {"label", "message"}.issubset(df.columns):
        labels, messages = df["label"], df["message"]
    else:
        raise ValueError("CSV must contain either v1/v2 or label/message columns.")

    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1}
    y = labels.astype(str).str.strip().str.lower().map(label_map)
    valid = y.notna() & messages.notna()
    messages = messages[valid].astype(str).reset_index(drop=True)
    y = y[valid].astype(int).reset_index(drop=True)
    if y.nunique() != 2:
        raise ValueError("The dataset must contain both ham and spam labels.")
    return messages, y


def evaluate(name: str, y_true, y_pred) -> dict:
    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"], zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpamShield models.")
    parser.add_argument("--skip-bert", action="store_true", help="Skip DistilBERT fine-tuning.")
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    messages, y = load_dataset(DATA_PATH)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        messages, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Fit TF-IDF on training text only to avoid leaking test-set vocabulary/statistics.
    X_train_clean = X_train_text.apply(preprocess_text)
    X_test_clean = X_test_text.apply(preprocess_text)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    X_train = tfidf.fit_transform(X_train_clean)
    X_test = tfidf.transform(X_test_clean)
    joblib.dump(tfidf, MODEL_DIR / "tfidf.pkl")

    # Oversampling is applied only to training data.
    ros = RandomOverSampler(random_state=90)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_res, y_train_res)
    joblib.dump(lr_model, MODEL_DIR / "logistic_model.pkl")
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)

    xgb_model = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, random_state=90,
        eval_metric="logloss", n_jobs=-1,
    )
    xgb_model.fit(X_train_res, y_train_res)
    joblib.dump(xgb_model, MODEL_DIR / "xgb_model.pkl")
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)

    results = {
        "logistic_regression": evaluate("Logistic Regression", y_test, lr_pred),
        "xgboost": evaluate("XGBoost", y_test, xgb_pred),
    }
    ensemble_probabilities = [lr_prob, xgb_prob]

    if not args.skip_bert:
        import torch
        from scipy.special import softmax
        from torch.utils.data import Dataset
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            Trainer,
            TrainingArguments,
        )

        class SpamDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels.astype(int).tolist()

            def __getitem__(self, idx):
                item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

            def __len__(self):
                return len(self.labels)

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        train_encodings = tokenizer(X_train_text.tolist(), truncation=True, padding=True, max_length=256)
        test_encodings = tokenizer(X_test_text.tolist(), truncation=True, padding=True, max_length=256)
        train_dataset = SpamDataset(train_encodings, y_train)
        test_dataset = SpamDataset(test_encodings, y_test)
        bert_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        training_args = TrainingArguments(
            output_dir=str(MODEL_DIR / "bert_training"),
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(model=bert_model, args=training_args, train_dataset=train_dataset)
        trainer.train()
        bert_model.save_pretrained(MODEL_DIR / "distilbert")
        tokenizer.save_pretrained(MODEL_DIR / "distilbert")
        predictions = trainer.predict(test_dataset).predictions
        bert_prob = softmax(predictions, axis=1)[:, 1]
        bert_pred = (bert_prob >= 0.5).astype(int)
        results["distilbert"] = evaluate("DistilBERT", y_test, bert_pred)
        ensemble_probabilities.append(bert_prob)

    ensemble_prob = np.mean(np.vstack(ensemble_probabilities), axis=0)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    results["ensemble"] = evaluate("Ensemble", y_test, ensemble_pred)
    (MODEL_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved model artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
