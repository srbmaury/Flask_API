# Offensive Content Detection API

A Flask API that classifies text as `Hateful Content`, `Offensive Content`, or
`Neither`. The model is trained from `labeled_data.csv` when the service starts.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_model.py
python predict.py
```

The API listens on `http://127.0.0.1:8000` by default.

Optional moderation thresholds are `FLAG_CONFIDENCE_THRESHOLD` (default
`0.60`) for full messages and `SEGMENT_FLAG_CONFIDENCE_THRESHOLD` (default
`0.55`) for offensive segments inside mixed-content text.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"You are an idiot"}'
```

## Deploy to Render

The included `render.yaml` defines a free web service with a health check. Link
this repository as a Render Blueprint, or use the equivalent build and start
commands from `render.yaml` in an existing service.
