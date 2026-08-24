import csv
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(os.environ.get("MODERATION_DATASET", BASE_DIR / "labeled_data.csv"))
MAX_TEXT_LENGTH = 10_000
CLASS_NAMES = {0: "Hateful Content", 1: "Offensive Content", 2: "Neither"}

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def load_training_data(path):
    texts, labels = [], []
    with path.open(encoding="utf-8", newline="") as dataset:
        for row in csv.DictReader(dataset):
            text = row.get("tweet", "").strip()
            try:
                label = int(row.get("class", ""))
            except ValueError:
                continue
            if text and label in CLASS_NAMES:
                texts.append(text)
                labels.append(label)
    if not texts:
        raise RuntimeError(f"No valid training rows found in {path}")
    return texts, labels


def train_model(path=DATASET_PATH):
    texts, labels = load_training_data(path)
    logger.info("Training moderation model from %s rows", len(texts))
    pipeline = Pipeline([
        ("features", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=75_000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        )),
    ])
    pipeline.fit(texts, labels)
    return pipeline


model = train_model()
app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        return jsonify({"error": "JSON body must include a text string"}), 400
    text = payload["text"].strip()
    if not text:
        return jsonify({"error": "text must not be empty"}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"text must not exceed {MAX_TEXT_LENGTH} characters"}), 413
    predicted_class = int(model.predict([text])[0])
    return jsonify({"prediction": CLASS_NAMES[predicted_class]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
