"""
Builds a small, REAL subgraph from the Elliptic TEST split, for the
frontend's "pick a sample subgraph" button and for quick API smoke-testing.
Saves RAW (unscaled) features -- the API applies the persisted scaler
itself, matching how any real caller would send data.

Run once after the CSVs are in place:
    python src/generate_sample_subgraph.py
Writes: api/sample_data/sample_subgraph.json
"""

import json
import os

import numpy as np

from data.elliptic_loader import get_raw_node_table, load_elliptic


def build_sample(num_illicit_seeds: int = 3, max_nodes: int = 25, seed: int = 42):
    # Reuse load_elliptic purely to get consistent masks/edge_index/labels;
    # we deliberately don't save a scaler here (this script isn't training).
    data, stats = load_elliptic(save_scaler_path=None)
    print(stats)

    rng = np.random.RandomState(seed)
    y = data.y.numpy()
    test_idx = np.where(data.test_mask.numpy())[0]
    illicit_test = [i for i in test_idx if y[i] == 1]

    if len(illicit_test) == 0:
        raise RuntimeError("No illicit test nodes found -- check the data split.")

    seeds = rng.choice(illicit_test, size=min(num_illicit_seeds, len(illicit_test)), replace=False)

    edge_index = data.edge_index.numpy()
    neighbors = {}
    for s, d in zip(edge_index[0], edge_index[1]):
        neighbors.setdefault(int(s), set()).add(int(d))

    selected = set(int(s) for s in seeds)
    for s in seeds:
        for n in list(neighbors.get(int(s), []))[:5]:  # cap fan-out per seed
            if len(selected) >= max_nodes:
                break
            selected.add(n)
        if len(selected) >= max_nodes:
            break
    selected = list(selected)[:max_nodes]
    selected_set = set(selected)

    # Raw (unscaled) features, via the SAME id<->index mapping load_elliptic
    # used internally -- this is exactly why get_raw_node_table exists,
    # rather than re-parsing the CSV here with separate logic that could drift.
    merged, feature_cols, id_to_idx, idx_to_id = get_raw_node_table()

    nodes = []
    for idx in selected:
        tx_id = idx_to_id[idx]
        row = merged.iloc[idx]
        raw_features = row[feature_cols].astype(float).tolist()
        nodes.append(
            {
                "id": str(tx_id),
                "features": raw_features,
                "true_label": "illicit" if y[idx] == 1 else "licit",
            }
        )

    edges = []
    seen_pairs = set()
    for s, d in zip(edge_index[0], edge_index[1]):
        s, d = int(s), int(d)
        if s in selected_set and d in selected_set:
            pair = tuple(sorted((s, d)))
            if pair not in seen_pairs:  # edges were symmetrized; de-dupe for display
                seen_pairs.add(pair)
                edges.append({"source": str(idx_to_id[s]), "target": str(idx_to_id[d])})

    sample = {"nodes": nodes, "edges": edges}

    out_dir = os.path.join("api", "sample_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_subgraph.json")
    with open(out_path, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Saved sample subgraph ({len(nodes)} nodes, {len(edges)} edges) to {out_path}")


if __name__ == "__main__":
    build_sample()