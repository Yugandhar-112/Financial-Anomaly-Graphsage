import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FinancialGraphSAGE(torch.nn.Module):
    """
    GraphSAGE classifier for transaction-graph fraud detection.

    Parameterized (num_layers, hidden_channels, dropout, aggr) so it can be
    driven entirely from configs/default.yaml in step 3 -- no hardcoded
    architecture choices left in code.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggr: str = "mean",
    ):
        super().__init__()
        assert num_layers >= 2, "Need at least 2 SAGEConv layers (input + output)."

        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    @torch.no_grad()
    def embed(self, x, edge_index):
        """Returns the penultimate-layer node embeddings (useful for viz/debugging)."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x