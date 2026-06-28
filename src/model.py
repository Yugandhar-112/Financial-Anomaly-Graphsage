import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FinancialGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FinancialGraphSAGE, self).__init__()
        # Layer 1: Neighborhood aggregation + self-loop projection
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        # Layer 2: Deeper structural embedding generation
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x