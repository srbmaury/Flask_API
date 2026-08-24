import csv
import logging
import os
import re
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(os.environ.get("MODERATION_DATASET", BASE_DIR / "labeled_data.csv"))
MODEL_PATH = Path(os.environ.get("MODERATION_MODEL", BASE_DIR / "moderation_model.joblib"))
MAX_TEXT_LENGTH = 10_000
FLAG_CONFIDENCE_THRESHOLD = float(os.environ.get("FLAG_CONFIDENCE_THRESHOLD", "0.60"))
SEGMENT_FLAG_CONFIDENCE_THRESHOLD = float(os.environ.get("SEGMENT_FLAG_CONFIDENCE_THRESHOLD", "0.55"))
CLASS_NAMES = {0: "Hateful Content", 1: "Offensive Content", 2: "Neither"}
CLEAR_CONDEMNATION = re.compile(
    r"^\s*(?:rape|racism|sexism|violence|abuse|harassment|hate(?: speech)?)\s+"
    r"(?:is|are)\s+(?:illegal|wrong|harmful|unacceptable|never okay)\s*[.!]?\s*$",
    re.IGNORECASE,
)

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


if MODEL_PATH.exists():
    logger.info("Loading moderation model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
else:
    logger.warning("Cached moderation model not found; training at startup")
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
    if CLEAR_CONDEMNATION.fullmatch(text):
        return jsonify({"prediction": "Neither", "confidence": 1.0})
    segments = [segment.strip() for segment in re.split(r"[.!?\n]+", text) if segment.strip()]
    candidates = [text, *segments] if len(segments) > 1 else [text]
    probability_rows = model.predict_proba(candidates)

    full_probabilities = probability_rows[0]
    full_winner = int(full_probabilities.argmax())
    predicted_class = int(model.classes_[full_winner])
    confidence = float(full_probabilities[full_winner])

    strongest_flag = None
    for probabilities in probability_rows:
        for class_id in (0, 1):
            class_index = list(model.classes_).index(class_id)
            score = float(probabilities[class_index])
            if strongest_flag is None or score > strongest_flag[1]:
                strongest_flag = (class_id, score)

    full_is_confident_flag = predicted_class != 2 and confidence >= FLAG_CONFIDENCE_THRESHOLD
    segment_is_confident_flag = strongest_flag and strongest_flag[1] >= SEGMENT_FLAG_CONFIDENCE_THRESHOLD
    if segment_is_confident_flag and (not full_is_confident_flag or strongest_flag[1] > confidence):
        predicted_class, confidence = strongest_flag
    elif not full_is_confident_flag:
        predicted_class = 2
    return jsonify({
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": round(confidence, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
