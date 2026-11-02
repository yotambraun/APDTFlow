"""Serve a fitted APDTFlow model over HTTP with FastAPI.

Train and save a model first (model.save("forecaster.pkl")), then:

    pip install apdtflow[serve]
    uvicorn serve_api:app --port 8000

    APDTFLOW_DEVICE=cuda uvicorn serve_api:app   # default: cpu

    curl localhost:8000/forecast
    curl -X POST localhost:8000/predict_when \
         -H 'Content-Type: application/json' \
         -d '{"threshold": 1.4, "direction": "below"}'
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from apdtflow import APDTFlowForecaster

MODEL_PATH = "forecaster.pkl"

app = FastAPI(title="APDTFlow forecast service")
# One window per request is latency-bound, and a CPU usually answers it faster
# than a GPU (see docs/gpu_and_performance.md); checkpoints load on any device.
model = APDTFlowForecaster.load(MODEL_PATH, device=os.environ.get("APDTFLOW_DEVICE", "cpu"))


class WhenRequest(BaseModel):
    threshold: float
    direction: str = "above"
    mode: str = "expected"
    alpha: float = 0.05


@app.get("/forecast")
def forecast():
    return {"forecast": model.predict().tolist()}


@app.post("/predict_when")
def predict_when(req: WhenRequest):
    r = model.predict_when(
        threshold=req.threshold, direction=req.direction,
        mode=req.mode, alpha=req.alpha,
    )
    return {
        "eta": str(r.eta), "earliest": str(r.earliest), "latest": str(r.latest),
        "act_by": str(r.act_by), "censored": r.censored,
        "low_confidence": r.low_confidence,
    }
