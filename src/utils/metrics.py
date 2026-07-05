"""
Shared evaluation utilities, factored out of the original train.py so the
GraphSAGE/GAT models, the MLP baseline, and the XGBoost baseline can all be
scored with EXACTLY the same threshold-optimization and metric logic --
otherwise a "the GNN wins" result could just be an artifact of the GNN
getting a more favorable metrics implementation.

`probs` and `actual` below are plain numpy arrays in all cases, so these
functions work identically whether the caller is a torch model or XGBoost.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score


def optimize_threshold(probs: np.ndarray, actual: np.ndarray, max_threshold: float = 0.5) -> float:
    """Finds the threshold that maximizes F1, searched only below max_threshold
    to prioritize recall on this recall-sensitive fraud-detection task."""
    precisions, recalls, thresholds = precision_recall_curve(actual, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    valid_idx = (thresholds <= max_threshold) & (precisions[:-1] > 0) & (recalls[:-1] > 0)
    if not np.any(valid_idx):
        return 0.35  # empirical balanced default for imbalanced graphs

    best_idx = np.where(valid_idx, f1_scores[:-1], 0).argmax()
    return float(thresholds[best_idx])


def calculate_metrics(probs: np.ndarray, actual: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(actual, probs) if len(set(actual)) > 1 else 0.5
    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc, "threshold": threshold}


def torch_logits_to_probs(logits, mask):
    """Helper for torch-based models: softmax logits -> positive-class probs,
    restricted to `mask`, returned as numpy for use with the functions above."""
    import torch

    probs = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    return probs