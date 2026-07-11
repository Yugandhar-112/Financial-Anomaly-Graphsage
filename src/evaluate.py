"""
Loads every trained checkpoint (GraphSAGE, GAT, MLP, XGBoost) and generates
the artifacts a reviewer actually wants to see:

  results/confusion_matrices.png   -- one subplot per model
  results/roc_comparison.png       -- all models overlaid on one ROC plot
  results/pr_comparison.png        -- all models overlaid on one PR plot
  results/metrics_summary.json     -- precision/recall/f1/auc per model,
                                       feeds directly into the README table

Run AFTER training all four models:
    python src/train.py
    python src/train.py --model_type gat
    python src/train.py --model_type mlp
    python src/models/xgboost_baseline.py
    python src/evaluate.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve

from data import load_dataset
from models import build_model
from utils.metrics import calculate_metrics, torch_logits_to_probs

RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
NEURAL_MODEL_TYPES = ["graphsage", "gat", "mlp"]


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_neural_model_probs(model_type: str, data):
    """Loads a torch checkpoint, rebuilds the exact architecture it was
    trained with, and returns (test_probs, test_actual, threshold)."""
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{model_type}_best.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[skip] No checkpoint found at {ckpt_path} -- train it first.")
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = build_model(checkpoint["model_config"], in_channels=checkpoint["in_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    probs = torch_logits_to_probs(logits, data.test_mask)
    actual = data.y[data.test_mask].cpu().numpy()
    return probs, actual, checkpoint["threshold"]


def get_xgboost_probs(data):
    """Loads the saved XGBoost model + threshold; XGBoost never sees
    edge_index at all, so this is a pure feature-based prediction."""
    model_path = os.path.join(CHECKPOINTS_DIR, "xgboost_best.json")
    meta_path = os.path.join(CHECKPOINTS_DIR, "xgboost_best_meta.json")
    if not (os.path.isfile(model_path) and os.path.isfile(meta_path)):
        print(f"[skip] No XGBoost checkpoint found at {model_path} -- train it first.")
        return None

    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    x = data.x.numpy()
    y = data.y.numpy()
    test_mask = data.test_mask.numpy()
    probs = model.predict_proba(x[test_mask])[:, 1]
    actual = y[test_mask]
    return probs, actual, meta["threshold"]


def collect_all_results(data) -> dict:
    """Returns {model_name: {"probs": ..., "actual": ..., "threshold": ...}}"""
    results = {}
    for model_type in NEURAL_MODEL_TYPES:
        out = get_neural_model_probs(model_type, data)
        if out is not None:
            probs, actual, threshold = out
            results[model_type] = {"probs": probs, "actual": actual, "threshold": threshold}

    xgb_out = get_xgboost_probs(data)
    if xgb_out is not None:
        probs, actual, threshold = xgb_out
        results["xgboost"] = {"probs": probs, "actual": actual, "threshold": threshold}

    return results


def plot_confusion_matrices(results: dict, out_path: str):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        preds = (r["probs"] >= r["threshold"]).astype(int)
        cm = confusion_matrix(r["actual"], preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Licit", "Illicit"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\n(threshold={r['threshold']:.3f})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_roc_comparison(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["actual"], r["probs"])
        auc = calculate_metrics(r["probs"], r["actual"], threshold=r["threshold"])["auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve -- Test Set (post-distribution-shift time steps)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pr_comparison(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(r["actual"], r["probs"])
        ax.plot(recall, precision, label=name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve -- Test Set")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_metrics_summary(results: dict, out_path: str):
    summary = {}
    for name, r in results.items():
        summary[name] = calculate_metrics(r["probs"], r["actual"], threshold=r["threshold"])
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")
    print("\n=== Summary (test set) ===")
    for name, m in summary.items():
        print(
            f"{name:10s} | P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} AUC={m['auc']:.4f}"
        )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = load_config()
    data_cfg = config["data"]

    data, stats = load_dataset(
        synthetic=False,
        data_dir=data_cfg["data_dir"],
        train_max_step=data_cfg.get("train_max_step", 29),
        val_max_step=data_cfg.get("val_max_step", 34),
    )
    print(stats)

    results = collect_all_results(data)
    if not results:
        print("No checkpoints found. Train at least one model before running evaluate.py.")
        return

    plot_confusion_matrices(results, os.path.join(RESULTS_DIR, "confusion_matrices.png"))
    plot_roc_comparison(results, os.path.join(RESULTS_DIR, "roc_comparison.png"))
    plot_pr_comparison(results, os.path.join(RESULTS_DIR, "pr_comparison.png"))
    save_metrics_summary(results, os.path.join(RESULTS_DIR, "metrics_summary.json"))


if __name__ == "__main__":
    main()