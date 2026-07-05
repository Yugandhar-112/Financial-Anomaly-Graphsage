import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FinancialGAT(torch.nn.Module):
    """
    Graph Attention Network variant, offered as a second graph-model option
    alongside GraphSAGE. Attention weights over neighbors can also make the
    model's decisions somewhat more interpretable than mean-aggregation
    GraphSAGE (which node in a transaction's neighborhood drove the score).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_layers >= 2

        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        # Final layer: average attention heads (concat=False) to land on out_channels
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x