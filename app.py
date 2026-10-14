from __future__ import annotations

from pathlib import Path
import re

import joblib
import nltk
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def ensure_nltk_resources() -> None:
    for resource_path, package in {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }.items():
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
    return " ".join(LEMMATIZER.lemmatize(word) for word in tokens)


app = FastAPI(title="SpamShield API", version="1.0.0", description="Classify SMS text as spam or ham.")

# Load the trained artifacts when the API starts.
tfidf_path = MODEL_DIR / "tfidf.pkl"
lr_path = MODEL_DIR / "logistic_model.pkl"
xgb_path = MODEL_DIR / "xgb_model.pkl"
if tfidf_path.exists() and lr_path.exists():
    tfidf = joblib.load(tfidf_path)
    lr_model = joblib.load(lr_path)
else:
    tfidf = lr_model = None
if xgb_path.exists():
    xgb_model = joblib.load(xgb_path)
else:
    xgb_model = None

bert_model = None
tokenizer = None
bert_path = MODEL_DIR / "distilbert"
if bert_path.exists():
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(bert_path))
    bert_model = DistilBertForSequenceClassification.from_pretrained(str(bert_path))
    bert_model.eval()


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {
            "logistic_regression": lr_model is not None,
            "xgboost": xgb_model is not None,
            "distilbert": bert_model is not None,
        },
    }


@app.post("/predict")
def predict(request: MessageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Please enter a message.")
    if tfidf is None or (lr_model is None and xgb_model is None and bert_model is None):
        raise HTTPException(
            status_code=503,
            detail="No trained model artifacts found. Run `python train_models.py` first.",
        )

    probabilities = {}
    if lr_model is not None and tfidf is not None:
        vector = tfidf.transform([preprocess_text(request.message)])
        probabilities["lr"] = float(lr_model.predict_proba(vector)[0, 1])
        if xgb_model is not None:
            probabilities["xgb"] = float(xgb_model.predict_proba(vector)[0, 1])
    elif xgb_model is not None:
        raise HTTPException(status_code=503, detail="TF-IDF artifacts are missing.")

    if bert_model is not None and tokenizer is not None:
        inputs = tokenizer(request.message, truncation=True, padding=True, max_length=256, return_tensors="pt")
        bert_model.eval()
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            probabilities["bert"] = float(torch.softmax(logits, dim=1)[0, 1].item())

    avg = sum(probabilities.values()) / len(probabilities)
    return {
        **probabilities,
        "avg": float(avg),
        "label": "SPAM" if avg >= 0.5 else "HAM",
        "threshold": 0.5,
        "models_used": list(probabilities.keys()),
    }


@app.get("/")
def analyzer():
    return FileResponse(ROOT / "spam_detector_gui.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse(ROOT / "dashboard.html")


# Keep this optional static mount for any extra local assets added later.
if (ROOT / "screenshots").exists():
    app.mount("/screenshots", StaticFiles(directory=ROOT / "screenshots"), name="screenshots")
