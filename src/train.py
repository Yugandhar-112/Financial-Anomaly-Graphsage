"""
Config-driven training entry point.

Usage:
    python src/train.py                                  # GraphSAGE on real Elliptic data
    python src/train.py --model_type gat                  # GAT instead
    python src/train.py --model_type mlp                   # non-graph baseline, same loop
    python src/train.py --synthetic                        # quick demo run, no CSVs needed
    python src/train.py --config configs/my_variant.yaml   # custom config file

All hyperparameters live in configs/default.yaml -- nothing here is hardcoded.
Every run is logged to MLflow (local mlruns/ folder by default; `mlflow ui`
to browse) with full config + per-epoch metrics + the final checkpoint as
an artifact, and is reproducible via the fixed seed in the config.
"""

import argparse
import copy
import os

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data import load_dataset
from models import build_model
from utils.metrics import calculate_metrics, optimize_threshold, torch_logits_to_probs
from utils.seed import set_seed


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten_config(cfg: dict, parent_key: str = "") -> dict:
    """Flattens nested config dict for MLflow's flat log_params API,
    e.g. {"model": {"type": "gat"}} -> {"model.type": "gat"}."""
    items = {}
    for k, v in cfg.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_config(v, key))
        else:
            items[key] = v
    return items


def train_pipeline(config: dict):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    data, stats = load_dataset(
        synthetic=data_cfg["synthetic"],
        data_dir=data_cfg["data_dir"],
        train_max_step=data_cfg.get("train_max_step", 29),
        val_max_step=data_cfg.get("val_max_step", 34),
    )
    data = data.to(device)

    model_cfg = config["model"]
    model = build_model(model_cfg, in_channels=data.num_features).to(device)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    criterion = FocalLoss(alpha=train_cfg["focal_alpha"], gamma=train_cfg["focal_gamma"])

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "financial-anomaly-graphsage"))
    run_name = mlflow_cfg.get("run_name") or f"{model_cfg['type']}_seed{config['seed']}"

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", f"{model_cfg['type']}_best.pt")

    best_val_f1 = -1.0
    best_state = None
    best_threshold = 0.5

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(config))
        if stats is not None:
            mlflow.log_params({f"data_stats.{k}": v for k, v in stats.__dict__.items()})

        print(f"Starting training | model={model_cfg['type']} | seed={config['seed']} | device={device}")

        for epoch in range(1, train_cfg["epochs"] + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % train_cfg["eval_every"] == 0 or epoch == train_cfg["epochs"]:
                model.eval()
                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)
                    val_probs = torch_logits_to_probs(val_out, data.val_mask)
                    val_actual = data.y[data.val_mask].cpu().numpy()
                    thresh = optimize_threshold(
                        val_probs, val_actual, max_threshold=train_cfg["threshold_search_max"]
                    )
                    val_metrics = calculate_metrics(val_probs, val_actual, threshold=thresh)

                print(
                    f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                    f"Threshold: {thresh:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                    f"Val AUC-ROC: {val_metrics['auc']:.4f}"
                )

                mlflow.log_metric("train_loss", loss.item(), step=epoch)
                mlflow.log_metric("val_f1", val_metrics["f1"], step=epoch)
                mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)
                mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)
                mlflow.log_metric("val_auc", val_metrics["auc"], step=epoch)
                mlflow.log_metric("val_threshold", thresh, step=epoch)

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_threshold = thresh
                    best_state = copy.deepcopy(model.state_dict())

        # Evaluate on test set using the checkpoint with the best VAL F1
        # (not just whatever the model looks like after the last epoch).
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            final_out = model(data.x, data.edge_index)
            test_probs = torch_logits_to_probs(final_out, data.test_mask)
            test_actual = data.y[data.test_mask].cpu().numpy()
            test_metrics = calculate_metrics(test_probs, test_actual, threshold=best_threshold)

        print("\n=== Test Performance (best val-F1 checkpoint) ===")
        print(f"Decision Threshold: {test_metrics['threshold']:.4f}")
        print(f"Precision:          {test_metrics['precision']:.4f}")
        print(f"Recall:             {test_metrics['recall']:.4f}")
        print(f"F1-Score:           {test_metrics['f1']:.4f}")
        print(f"AUC-ROC:            {test_metrics['auc']:.4f}")

        mlflow.log_metric("test_precision", test_metrics["precision"])
        mlflow.log_metric("test_recall", test_metrics["recall"])
        mlflow.log_metric("test_f1", test_metrics["f1"])
        mlflow.log_metric("test_auc", test_metrics["auc"])

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": model_cfg["type"],
                "in_channels": data.num_features,
                "model_config": model_cfg,
                "threshold": best_threshold,
                "test_metrics": test_metrics,
            },
            checkpoint_path,
        )
        mlflow.log_artifact(checkpoint_path)
        print(f"\nSaved checkpoint to {checkpoint_path}")

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Override config: use synthetic demo data")
    parser.add_argument("--model_type", type=str, default=None, choices=["graphsage", "gat", "mlp"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        cfg["data"]["synthetic"] = True
    if args.model_type:
        cfg["model"]["type"] = args.model_type

    train_pipeline(cfg)