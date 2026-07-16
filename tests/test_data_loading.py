"""
Sanity tests for the Elliptic data loading pipeline. Builds a small synthetic
dataset on disk (matching the real schema) rather than requiring the actual
Kaggle CSVs, so these tests run in CI without the real dataset present.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.elliptic_loader import load_elliptic, get_raw_node_table

NUM_FEATURES = 5


def _write_synthetic_dataset(data_dir):
    """
    12 nodes spanning time steps 1-6, two per step:
      - steps 1-3 (6 nodes)  -> train
      - step 4     (2 nodes) -> val
      - steps 5-6 (4 nodes)  -> test
    A couple of nodes are left 'unknown' to check mask exclusion.
    """
    rng = np.random.default_rng(0)

    tx_ids = list(range(100, 112))
    time_steps = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    classes = ["1", "2", "1", "2", "1", "2", "unknown", "2", "1", "unknown", "1", "2"]

    feature_rows = []
    for tx, ts in zip(tx_ids, time_steps):
        feature_rows.append([tx, ts] + list(rng.normal(size=NUM_FEATURES)))
    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(data_dir / "elliptic_txs_features.csv", header=False, index=False)

    classes_df = pd.DataFrame({"txId": tx_ids, "class": classes})
    classes_df.to_csv(data_dir / "elliptic_txs_classes.csv", index=False)

    edges_df = pd.DataFrame({
        "txId1": [100, 102, 104, 106, 108, 110],
        "txId2": [101, 103, 105, 107, 109, 111],
    })
    edges_df.to_csv(data_dir / "elliptic_txs_edgelist.csv", index=False)


@pytest.fixture
def synthetic_data_dir(tmp_path):
    _write_synthetic_dataset(tmp_path)
    return tmp_path


def test_load_elliptic_shapes(synthetic_data_dir):
    data, stats = load_elliptic(
        data_dir=str(synthetic_data_dir),
        train_max_step=3,
        val_max_step=4,
        save_scaler_path=None,
    )

    assert data.x.shape == (12, NUM_FEATURES)
    assert data.y.shape == (12,)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 12  # undirected doubles the 6 input edges

    assert stats.num_nodes == 12
    assert stats.num_features == NUM_FEATURES


def test_load_elliptic_masks_are_disjoint_and_time_based(synthetic_data_dir):
    data, stats = load_elliptic(
        data_dir=str(synthetic_data_dir),
        train_max_step=3,
        val_max_step=4,
        save_scaler_path=None,
    )

    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx = data.val_mask.nonzero(as_tuple=True)[0]
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]

    assert set(train_idx.tolist()) & set(val_idx.tolist()) == set()
    assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()
    assert set(val_idx.tolist()) & set(test_idx.tolist()) == set()

    assert (data.time_step[train_idx] <= 3).all()
    assert ((data.time_step[val_idx] > 3) & (data.time_step[val_idx] <= 4)).all()
    assert (data.time_step[test_idx] > 4).all()


def test_load_elliptic_excludes_unknown_labels_from_all_masks(synthetic_data_dir):
    data, stats = load_elliptic(
        data_dir=str(synthetic_data_dir),
        train_max_step=3,
        val_max_step=4,
        save_scaler_path=None,
    )

    unknown_idx = (data.y == -1).nonzero(as_tuple=True)[0]
    assert unknown_idx.numel() > 0

    any_mask = data.train_mask | data.val_mask | data.test_mask
    assert not any_mask[unknown_idx].any()


def test_load_elliptic_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_elliptic(data_dir=str(tmp_path), save_scaler_path=None)


def test_get_raw_node_table_matches_load_elliptic_ordering(synthetic_data_dir):
    merged, feature_cols, id_to_idx, idx_to_id = get_raw_node_table(str(synthetic_data_dir))

    assert len(feature_cols) == NUM_FEATURES
    assert len(id_to_idx) == 12

    for tx_id, idx in id_to_idx.items():
        assert idx_to_id[idx] == tx_id
