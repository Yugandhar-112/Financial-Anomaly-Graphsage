"""
Synthetic transaction graph generator.

This is the ORIGINAL data source from the learning-exercise version of this
project. It is kept only as an opt-in `--synthetic` demo/test mode: useful for
fast CI runs, architecture sanity checks, and offline demos when the real
Elliptic CSVs aren't available -- it is NOT used for any reported results.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def create_mock_transaction_data(num_nodes=1000, num_edges=4000, seed=42):
    """Generates structured synthetic financial transaction graphs with homophily."""
    rng = np.random.RandomState(seed)

    # 10 continuous transaction attributes
    features = rng.normal(loc=0.0, scale=1.0, size=(num_nodes, 10))

    # Imbalanced targets: 0 = Legit (95%), 1 = Anomaly (5%)
    labels = rng.choice([0, 1], size=num_nodes, p=[0.95, 0.05])

    # Force anomalous nodes to exhibit distinctive feature shifts (fraud signatures)
    anomaly_indices = np.where(labels == 1)[0]
    features[anomaly_indices] += rng.normal(
        loc=1.5, scale=0.5, size=(len(anomaly_indices), 10)
    )

    edge_src, edge_dst = [], []
    for _ in range(num_edges):
        if rng.rand() < 0.4 and len(anomaly_indices) > 1:
            # 40% chance to force an anomaly-to-anomaly coordination edge
            src = rng.choice(anomaly_indices)
            dst = rng.choice(anomaly_indices)
            while src == dst:
                dst = rng.choice(anomaly_indices)
        else:
            # Standard random background transaction flows
            src = rng.randint(0, num_nodes)
            dst = rng.randint(0, num_nodes)
        edge_src.append(src)
        edge_dst.append(dst)

    edge_index = np.stack([edge_src, edge_dst], axis=0)
    return features, labels, edge_index


def get_synthetic_dataset(num_nodes=1000, num_edges=4000, seed=42):
    raw_features, raw_labels, edge_index = create_mock_transaction_data(
        num_nodes=num_nodes, num_edges=num_edges, seed=seed
    )

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(raw_features)

    x = torch.tensor(normalized_features, dtype=torch.float)
    y = torch.tensor(raw_labels, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    num_nodes = x.size(0)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_nodes)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


if __name__ == "__main__":
    data = get_synthetic_dataset()
    print(data)