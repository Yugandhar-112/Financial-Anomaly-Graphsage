import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def create_mock_transaction_data(num_nodes=1000, num_edges=3000):
    np.random.seed(42)
    features = np.random.randn(num_nodes, 10)
    labels = np.random.choice([0, 1], size=num_nodes, p=[0.95, 0.05])
    
    edge_src = np.random.randint(0, num_nodes, size=num_edges)
    edge_dst = np.random.randint(0, num_nodes, size=num_edges)
    edge_index = np.stack([edge_src, edge_dst], axis=0)
    
    return features, labels, edge_index

def get_financial_dataset():
    raw_features, raw_labels, edge_index = create_mock_transaction_data()
    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(raw_features)
    
    x = torch.tensor(normalized_features, dtype=torch.float)
    y = torch.tensor(raw_labels, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    num_nodes = x.size(0)
    indices = np.random.permutation(num_nodes)
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