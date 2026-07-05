import torch
import torch.nn.functional as F


class NodeFeatureMLP(torch.nn.Module):
    """
    Non-graph baseline: classifies each transaction using ONLY its own
    feature vector, with no message passing over the graph at all.

    Deliberately shares the `forward(x, edge_index)` signature with the
    graph models (edge_index is accepted but ignored) so it can be trained
    and evaluated through the exact same loop, on the exact same
    train/val/test masks -- making the baseline-vs-GNN comparison in the
    README a true apples-to-apples test of "how much does graph structure
    actually help".
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_layers >= 2

        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x, edge_index=None):  # edge_index ignored on purpose
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x