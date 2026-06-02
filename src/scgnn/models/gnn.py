"""
GraphSAGE and GAT models for temporal contagion prediction.

Each model takes a PyG Data/Batch object with:
  - x: node features  (N, F)
  - edge_index: (2, E)
  - edge_attr: edge features (E, D)  [optional]
and returns per-node logits (N, 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class GraphSAGEContagion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.head = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return self.head(x)


class GATContagion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        edge_dim: int = 0,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        _edge_dim = edge_dim if edge_dim > 0 else None
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, edge_dim=_edge_dim, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=_edge_dim, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, edge_dim=_edge_dim, dropout=dropout))
        self.head = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        return self.head(x)
