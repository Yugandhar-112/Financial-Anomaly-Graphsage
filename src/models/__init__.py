from .graphsage import FinancialGraphSAGE
from .gat import FinancialGAT
from .baseline_mlp import NodeFeatureMLP

__all__ = ["FinancialGraphSAGE", "FinancialGAT", "NodeFeatureMLP", "build_model"]


def build_model(model_cfg: dict, in_channels: int):
    """Single factory so train.py just does `build_model(config['model'], in_channels)`
    and never hardcodes which class to instantiate."""
    model_type = model_cfg["type"]

    if model_type == "graphsage":
        return FinancialGraphSAGE(
            in_channels=in_channels,
            hidden_channels=model_cfg["hidden_channels"],
            out_channels=model_cfg["out_channels"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            aggr=model_cfg.get("aggr", "mean"),
        )
    elif model_type == "gat":
        return FinancialGAT(
            in_channels=in_channels,
            hidden_channels=model_cfg["hidden_channels"],
            out_channels=model_cfg["out_channels"],
            num_layers=model_cfg["num_layers"],
            heads=model_cfg.get("heads", 4),
            dropout=model_cfg["dropout"],
        )
    elif model_type == "mlp":
        return NodeFeatureMLP(
            in_channels=in_channels,
            hidden_channels=model_cfg["hidden_channels"],
            out_channels=model_cfg["out_channels"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Expected graphsage | gat | mlp.")