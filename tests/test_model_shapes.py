"""
Shape/sanity tests for all three PyG-based models built via build_model().
No real data or GPU required -- uses small synthetic tensors.
"""
import pytest
import torch

from src.models import build_model, FinancialGraphSAGE

IN_CHANNELS = 8
HIDDEN = 16
OUT_CHANNELS = 2
NUM_NODES = 20
NUM_EDGES = 40


def _make_dummy_graph():
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long)
    return x, edge_index


@pytest.mark.parametrize("model_type", ["graphsage", "gat", "mlp"])
def test_build_model_forward_shape(model_type):
    model_cfg = {
        "type": model_type,
        "hidden_channels": HIDDEN,
        "out_channels": OUT_CHANNELS,
        "num_layers": 2,
        "dropout": 0.3,
    }
    model = build_model(model_cfg, in_channels=IN_CHANNELS)
    x, edge_index = _make_dummy_graph()

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)

    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_build_model_unknown_type_raises():
    with pytest.raises(ValueError):
        build_model({"type": "not_a_real_model"}, in_channels=IN_CHANNELS)


def test_graphsage_requires_at_least_two_layers():
    with pytest.raises(AssertionError):
        FinancialGraphSAGE(in_channels=IN_CHANNELS, num_layers=1)


def test_graphsage_embed_returns_penultimate_layer_shape():
    model = FinancialGraphSAGE(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN,
        out_channels=OUT_CHANNELS,
        num_layers=2,
    )
    x, edge_index = _make_dummy_graph()
    embeddings = model.embed(x, edge_index)
    assert embeddings.shape == (NUM_NODES, HIDDEN)


def test_graphsage_three_layers_forward_shape():
    model_cfg = {
        "type": "graphsage",
        "hidden_channels": HIDDEN,
        "out_channels": OUT_CHANNELS,
        "num_layers": 3,
        "dropout": 0.3,
    }
    model = build_model(model_cfg, in_channels=IN_CHANNELS)
    x, edge_index = _make_dummy_graph()
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
