"""
FastAPI inference service for the Financial Anomaly Detection project.

Run locally:
    uvicorn main:app --reload --app-dir api

Endpoints:
    GET  /health                -> liveness check
    GET  /sample-subgraph       -> a small real subgraph from the Elliptic
                                    test set, for the frontend's demo button
    POST /predict/subgraph      -> score a subgraph (single transaction +
                                    its neighborhood, or a larger uploaded
                                    subgraph) with a chosen model
"""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import get_model, predict_subgraph
from schemas import PredictionResult, SubgraphRequest, SubgraphResponse

app = FastAPI(
    title="Financial Anomaly Detection API",
    description=(
        "Serves GraphSAGE / GAT / MLP / XGBoost fraud-detection models "
        "trained on the Elliptic Bitcoin transaction graph dataset."
    ),
    version="1.0.0",
)

# Configure via env var in deployment (e.g. FRONTEND_ORIGIN=https://your-app.vercel.app).
# Defaults to "*" for local development only.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_SUBGRAPH_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "sample_subgraph.json")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sample-subgraph")
def sample_subgraph():
    """Returns a small REAL subgraph from the Elliptic test set (with ground-truth
    labels included for display) that the frontend can render or feed straight
    into /predict/subgraph."""
    if not os.path.isfile(SAMPLE_SUBGRAPH_PATH):
        raise HTTPException(
            status_code=404,
            detail="Sample subgraph not generated yet. Run `python src/generate_sample_subgraph.py`.",
        )
    with open(SAMPLE_SUBGRAPH_PATH) as f:
        return json.load(f)


@app.post("/predict/subgraph", response_model=SubgraphResponse)
def predict(payload: SubgraphRequest):
    try:
        predictions, threshold = predict_subgraph(
            nodes=[n.dict() for n in payload.nodes],
            edges=[e.dict() for e in payload.edges],
            model_type=payload.model_type,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SubgraphResponse(
        predictions=[PredictionResult(**p) for p in predictions],
        threshold=threshold,
        model_type=payload.model_type,
    )


@app.on_event("startup")
def preload_default_model():
    # Best-effort warm start so the first real request isn't slow; failures
    # here are fine (e.g. checkpoint not trained yet) -- predict() will raise
    # a clear 503 for the specific model_type requested instead.
    try:
        get_model("graphsage")
    except FileNotFoundError:
        pass