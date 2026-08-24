import joblib

from predict import MODEL_PATH, model


MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH, compress=3)
print(f"Saved moderation model to {MODEL_PATH}")
