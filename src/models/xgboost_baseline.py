"""
XGBoost baseline: classifies transactions using ONLY node features, with no
graph structure at all -- not even the "ignore edge_index" trick the MLP
baseline uses, since XGBoost has no concept of message passing to begin with.

Uses the exact same time-based train/val/test masks as the GNN, and the same
metrics/threshold-optimization utilities (src/utils/metrics.py), so its
numbers are directly comparable in the README's baseline-vs-GNN table.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from data import load_dataset  # noqa: E402
from utils.metrics import calculate_metrics, optimize_threshold  # noqa: E402


def run_xgboost_baseline(data_dir: str = "data/raw/elliptic", synthetic: bool = False, seed: int = 42):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "xgboost is not installed. Add it via `pip install xgboost` "
            "(also pinned in requirements.txt)."
        ) from e

    data, stats = load_dataset(synthetic=synthetic, data_dir=data_dir)
    if stats is not None:
        print(stats)

    x = data.x.numpy()
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    # Same imbalance-handling spirit as the focal loss used for the GNN:
    # up-weight the minority (illicit) class rather than leaving it as-is.
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / max(num_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_probs = model.predict_proba(x_val)[:, 1]
    best_thresh = optimize_threshold(val_probs, y_val)

    test_probs = model.predict_proba(x_test)[:, 1]
    metrics = calculate_metrics(test_probs, y_test, threshold=best_thresh)

    print("\n=== XGBoost Baseline (no graph structure) -- Test Performance ===")
    print(f"Decision Threshold: {metrics['threshold']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1-Score:           {metrics['f1']:.4f}")
    print(f"AUC-ROC:            {metrics['auc']:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    model.save_model("checkpoints/xgboost_best.json")
    with open("checkpoints/xgboost_best_meta.json", "w") as f:
        json.dump({"threshold": metrics["threshold"], "test_metrics": metrics}, f, indent=2)
    print("Saved checkpoint to checkpoints/xgboost_best.json (+ _meta.json)")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/elliptic")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run_xgboost_baseline(data_dir=args.data_dir, synthetic=args.synthetic)