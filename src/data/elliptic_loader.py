"""
Elliptic Bitcoin transaction graph loader.

Dataset: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Citation: Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with
Graph Convolutional Networks for Financial Forensics", KDD 2019 Workshop.

Expected files (place under `data_dir`, default data/raw/elliptic/):
  - elliptic_txs_features.csv   (no header: txId, time_step, 165 feature cols)
  - elliptic_txs_classes.csv    (header: txId, class -> '1'=illicit, '2'=licit, 'unknown')
  - elliptic_txs_edgelist.csv   (header: txId1, txId2)

Label convention used downstream (binary fraud detection):
  1 = illicit (fraud)   0 = licit   -1 = unknown (unlabeled, excluded from masks)

Split strategy: TIME-BASED, matching the original Elliptic paper. The 49 time
steps are split chronologically so evaluation reflects a realistic deployment
scenario (train on the past, detect fraud in the future) rather than a random
shuffle, which would leak future graph structure into training.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGELIST_FILE = "elliptic_txs_edgelist.csv"

# Original paper split point: train on steps 1-34, test on steps 35-49.
# We additionally carve the last few train-range steps out as validation.
DEFAULT_TRAIN_MAX_STEP = 29   # time_step <= 29           -> train
DEFAULT_VAL_MAX_STEP = 34     # 29 < time_step <= 34      -> validation
# time_step > 34                                          -> test


@dataclass
class EllipticStats:
    num_nodes: int
    num_edges: int
    num_features: int
    num_labeled: int
    num_illicit: int
    num_licit: int
    num_unknown: int
    num_train: int
    num_val: int
    num_test: int
    time_step_range: tuple


def _check_files_exist(data_dir: str):
    missing = []
    for fname in (FEATURES_FILE, CLASSES_FILE, EDGELIST_FILE):
        if not os.path.isfile(os.path.join(data_dir, fname)):
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            f"Missing Elliptic file(s) in '{data_dir}': {missing}. "
            f"Download the dataset from "
            f"https://www.kaggle.com/datasets/ellipticco/elliptic-data-set "
            f"and place the CSVs (unrenamed) in that directory."
        )


def load_elliptic(
    data_dir: str = "data/raw/elliptic",
    train_max_step: int = DEFAULT_TRAIN_MAX_STEP,
    val_max_step: int = DEFAULT_VAL_MAX_STEP,
    make_undirected: bool = True,
) -> tuple:
    """
    Loads the Elliptic dataset and builds a PyG Data object with time-based
    train/val/test masks over the labeled subset of nodes.

    Returns:
        data: torch_geometric.data.Data with x, y, edge_index, and
              train_mask / val_mask / test_mask (bool tensors over ALL nodes;
              unknown-labeled nodes are False in every mask but still
              participate in message passing).
        stats: EllipticStats summary for logging/README numbers.
    """
    _check_files_exist(data_dir)

    # --- features (no header) ---
    feat_path = os.path.join(data_dir, FEATURES_FILE)
    features_df = pd.read_csv(feat_path, header=None)
    n_cols = features_df.shape[1]
    col_names = ["txId", "time_step"] + [f"feat_{i}" for i in range(n_cols - 2)]
    features_df.columns = col_names

    # --- classes ---
    classes_path = os.path.join(data_dir, CLASSES_FILE)
    classes_df = pd.read_csv(classes_path)
    classes_df["txId"] = classes_df["txId"].astype(features_df["txId"].dtype)
    label_map = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}
    classes_df["label"] = classes_df["class"].map(label_map)
    if classes_df["label"].isna().any():
        bad = classes_df.loc[classes_df["label"].isna(), "class"].unique()
        raise ValueError(f"Unrecognized class value(s) in classes file: {bad}")

    merged = features_df.merge(classes_df[["txId", "label"]], on="txId", how="left")
    merged["label"] = merged["label"].fillna(-1).astype(int)

    # --- contiguous node index mapping ---
    tx_ids = merged["txId"].values
    id_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
    num_nodes = len(tx_ids)

    feature_cols = [c for c in merged.columns if c.startswith("feat_")]
    raw_x = merged[feature_cols].values.astype(np.float32)
    time_steps = merged["time_step"].values.astype(int)
    y_np = merged["label"].values.astype(int)

    # --- edges ---
    edge_path = os.path.join(data_dir, EDGELIST_FILE)
    edges_df = pd.read_csv(edge_path)
    edges_df = edges_df[
        edges_df["txId1"].isin(id_to_idx) & edges_df["txId2"].isin(id_to_idx)
    ]
    src = edges_df["txId1"].map(id_to_idx).values
    dst = edges_df["txId2"].map(id_to_idx).values

    if make_undirected:
        edge_index_np = np.stack(
            [np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0
        )
    else:
        edge_index_np = np.stack([src, dst], axis=0)

    # --- time-based masks (only over labeled nodes) ---
    labeled_mask_np = y_np != -1
    train_mask_np = labeled_mask_np & (time_steps <= train_max_step)
    val_mask_np = labeled_mask_np & (time_steps > train_max_step) & (time_steps <= val_max_step)
    test_mask_np = labeled_mask_np & (time_steps > val_max_step)

    # --- feature scaling: fit ONLY on train nodes to avoid leakage ---
    scaler = StandardScaler()
    scaler.fit(raw_x[train_mask_np])
    x_np = scaler.transform(raw_x)

    x = torch.tensor(x_np, dtype=torch.float)
    y = torch.tensor(y_np, dtype=torch.long)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask_np, dtype=torch.bool)
    data.time_step = torch.tensor(time_steps, dtype=torch.long)

    stats = EllipticStats(
        num_nodes=num_nodes,
        num_edges=edge_index.size(1),
        num_features=x.size(1),
        num_labeled=int(labeled_mask_np.sum()),
        num_illicit=int((y_np == 1).sum()),
        num_licit=int((y_np == 0).sum()),
        num_unknown=int((y_np == -1).sum()),
        num_train=int(train_mask_np.sum()),
        num_val=int(val_mask_np.sum()),
        num_test=int(test_mask_np.sum()),
        time_step_range=(int(time_steps.min()), int(time_steps.max())),
    )

    return data, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check the Elliptic loader")
    parser.add_argument("--data_dir", type=str, default="data/raw/elliptic")
    args = parser.parse_args()

    data, stats = load_elliptic(data_dir=args.data_dir)
    print(data)
    print(stats)