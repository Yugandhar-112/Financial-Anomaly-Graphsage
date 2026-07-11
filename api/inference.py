"""
Loads trained checkpoints (cached in-process after first use) and scores
transaction subgraphs. Applies the SAME StandardScaler fit during training
to any incoming raw feature vectors, so predictions here match what
src/evaluate.py reported.
"""

import json
import os
import sys

import joblib
import numpy as np
import torch

# Make the training-side `src/` package importable (models.build_model etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import build_model  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
SCALER_PATH = os.path.join(CHECKPOINTS_DIR, "feature_scaler.joblib")

VALID_MODEL_TYPES = ("graphsage", "gat", "mlp", "xgboost")

_model_cache = {}
_scaler_cache = {"scaler": None}


def _get_scaler():
    if _scaler_cache["scaler"] is None:
        if not os.path.isfile(SCALER_PATH):
            raise FileNotFoundError(
                f"Feature scaler not found at {SCALER_PATH}. Run `python src/train.py` "
                "at least once from the repo root (fitting + saving the scaler is a "
                "side effect of loading the real Elliptic data)."
            )
        _scaler_cache["scaler"] = joblib.load(SCALER_PATH)
    return _scaler_cache["scaler"]


def _load_torch_model(model_type: str):
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{model_type}_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train this model first.")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = build_model(checkpoint["model_config"], in_channels=checkpoint["in_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["threshold"]


def _load_xgboost_model():
    import xgboost as xgb

    model_path = os.path.join(CHECKPOINTS_DIR, "xgboost_best.json")
    meta_path = os.path.join(CHECKPOINTS_DIR, "xgboost_best_meta.json")
    if not (os.path.isfile(model_path) and os.path.isfile(meta_path)):
        raise FileNotFoundError(f"No XGBoost checkpoint at {model_path}. Train it first.")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta["threshold"]


def get_model(model_type: str):
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected one of {VALID_MODEL_TYPES}.")
    if model_type not in _model_cache:
        if model_type == "xgboost":
            _model_cache[model_type] = _load_xgboost_model()
        else:
            _model_cache[model_type] = _load_torch_model(model_type)
    return _model_cache[model_type]


def predict_subgraph(nodes: list, edges: list, model_type: str = "graphsage"):
    """
    nodes: list of {"id": str, "features": List[float]} with RAW (unscaled) features
    edges: list of {"source": str, "target": str}

    Returns (predictions, threshold) where predictions is a list of
    {"id":..., "fraud_probability":..., "flagged": bool}.
    """
    if len(nodes) == 0:
        raise ValueError("At least one node is required.")

    scaler = _get_scaler()
    model, threshold = get_model(model_type)

    ids = [n["id"] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    raw_x = np.array([n["features"] for n in nodes], dtype=np.float32)

    expected_dim = scaler.mean_.shape[0]
    if raw_x.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} features per node (Elliptic feature vector "
            f"length), got {raw_x.shape[1]}."
        )

    x_scaled = scaler.transform(raw_x).astype(np.float32)

    if model_type == "xgboost":
        probs = model.predict_proba(x_scaled)[:, 1]
    else:
        src, dst = [], []
        for e in edges:
            if e["source"] in id_to_idx and e["target"] in id_to_idx:
                s, t = id_to_idx[e["source"]], id_to_idx[e["target"]]
                src += [s, t]
                dst += [t, s]  # symmetrize, matching how training data was built

        edge_index = (
            torch.tensor([src, dst], dtype=torch.long)
            if src
            else torch.zeros((2, 0), dtype=torch.long)
        )

        x_tensor = torch.tensor(x_scaled, dtype=torch.float)
        with torch.no_grad():
            logits = model(x_tensor, edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    predictions = [
        {"id": nid, "fraud_probability": float(p), "flagged": bool(p >= threshold)}
        for nid, p in zip(ids, probs)
    ]
    return predictions, threshold