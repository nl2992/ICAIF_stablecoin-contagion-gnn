"""
Plan E — TGN-lite: Temporal Graph Network variant.

Reuses GAT message-passing but feeds each node a GRU hidden state initialised
from the previous timestep's node embedding.  This allows the model to capture
temporal memory across hourly snapshots within a crisis episode.

Architecture:
  input_proj : Linear(in_channels + memory_dim, hidden_channels)
  GAT layers : same as GATContagion (3 layers, 4 heads)
  gru_cell   : GRUCell(hidden_channels, memory_dim)  — updates per-node memory
  head       : Linear(hidden_channels, 1)

Key difference from static GAT:
  - memory (N, memory_dim) is threaded across snapshots within an episode
  - at snapshot t=0, memory is zeros (no prior context)
  - at snapshot t>0, memory = gru_cell(embedding_t-1, memory_t-1)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from scgnn.data.dataset import load_episode, to_pyg_snapshots


class TGNLite(nn.Module):
    """GAT with per-node GRU memory across timesteps."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        edge_dim: int = 0,
        memory_dim: int = 32,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_channels = hidden_channels
        _edge_dim = edge_dim if edge_dim > 0 else None

        # Project [node_feat || memory] into the GAT input dimension
        self.input_proj = nn.Linear(in_channels + memory_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(hidden_channels, hidden_channels, heads=heads,
                    edge_dim=_edge_dim, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads,
                        edge_dim=_edge_dim, dropout=dropout))
        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False,
                    edge_dim=_edge_dim, dropout=dropout))

        # GRU cell for memory update (one per node independently)
        self.gru_cell = nn.GRUCell(hidden_channels, memory_dim)
        self.head = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (N, in_channels) node features
            edge_index: (2, E)
            edge_attr:  (E, edge_dim) or None
            memory:     (N, memory_dim) or None (zeros if first snapshot)
        Returns:
            logits:     (N, 1)
            new_memory: (N, memory_dim)
        """
        N = x.size(0)
        if memory is None:
            memory = torch.zeros(N, self.memory_dim, device=x.device, dtype=x.dtype)

        h = F.elu(self.input_proj(torch.cat([x, memory], dim=-1)))

        for conv in self.convs[:-1]:
            h = F.elu(conv(h, edge_index, edge_attr=edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index, edge_attr=edge_attr)

        new_memory = self.gru_cell(h, memory)
        return self.head(h), new_memory


class TGNContagionTrainer:
    """
    Trainer for TGN-lite that threads memory across snapshots within each episode.
    Implements the same public interface as GNNContagionTrainer.
    """

    def __init__(
        self,
        horizon: int = 60,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 100,
        patience: int = 12,
        seed: int = 42,
        device: str = "cpu",
        memory_dim: int = 32,
    ):
        self.horizon = horizon
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = device
        self.memory_dim = memory_dim
        self.model: Optional[TGNLite] = None
        self.in_dim: Optional[int] = None

    def _load_episode_data(self, names: List[str]) -> list:
        """Load all snapshots per episode, preserving temporal order within each episode."""
        episodes = []
        for name in names:
            b = load_episode(name)
            snaps = []
            for si, d in to_pyg_snapshots(b, self.horizon):
                if d.eval_mask.sum() == 0:
                    continue
                snaps.append(d.to(self.device))
            if snaps:
                episodes.append(snaps)
        return episodes

    def fit(self, train_names: List[str], val_names: Optional[List[str]] = None) -> "TGNContagionTrainer":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_episodes = self._load_episode_data(train_names)
        if not train_episodes:
            raise RuntimeError("No trainable snapshots")

        self.in_dim = train_episodes[0][0].x.shape[1]
        edge_dim = train_episodes[0][0].edge_attr.shape[1] if train_episodes[0][0].edge_attr is not None else 0

        self.model = TGNLite(
            in_channels=self.in_dim,
            hidden_channels=self.hidden,
            heads=4,
            num_layers=self.layers,
            dropout=self.dropout,
            edge_dim=edge_dim,
            memory_dim=self.memory_dim,
        ).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Class weight from train labels
        pos = sum(float(d.y[d.eval_mask].sum()) for ep in train_episodes for d in ep)
        tot = sum(int(d.eval_mask.sum()) for ep in train_episodes for d in ep)
        pos_weight = torch.tensor([(tot - pos) / max(pos, 1.0)], device=self.device)

        val_episodes = self._load_episode_data(val_names) if val_names else None
        best_val, best_state, bad = float("inf"), None, 0

        for epoch in range(self.epochs):
            self.model.train()
            np.random.shuffle(train_episodes)
            for ep_snaps in train_episodes:
                memory = None
                for d in ep_snaps:
                    opt.zero_grad()
                    logits, memory = self.model(d.x, d.edge_index, d.edge_attr, memory)
                    memory = memory.detach()  # detach to avoid BPTT across snapshots
                    loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(-1)[d.eval_mask],
                        d.y[d.eval_mask],
                        pos_weight=pos_weight,
                    )
                    loss.backward()
                    opt.step()

            ref = val_episodes if val_episodes else train_episodes
            vloss = self._epoch_loss(ref, pos_weight)
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _epoch_loss(self, episodes: list, pos_weight: torch.Tensor) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for ep_snaps in episodes:
            memory = None
            for d in ep_snaps:
                logits, memory = self.model(d.x, d.edge_index, d.edge_attr, memory)
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1)[d.eval_mask],
                    d.y[d.eval_mask],
                    pos_weight=pos_weight,
                )
                tot += float(loss) * int(d.eval_mask.sum())
                n += int(d.eval_mask.sum())
        return tot / max(n, 1)

    @torch.no_grad()
    def predict_episodes(self, names: List[str]) -> np.ndarray:
        """Return probabilities in tabular_from_episodes row order."""
        self.model.eval()
        probs: list[float] = []
        for name in names:
            b = load_episode(name)
            node_strs = b["node_strs"]
            origin = b["origin"]
            active = b["active"]
            memory = None
            for si, d in to_pyg_snapshots(b, self.horizon):
                d = d.to(self.device)
                logits, memory = self.model(d.x, d.edge_index, d.edge_attr, memory)
                p = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                for j in range(len(node_strs)):
                    if not active[si, j] or node_strs[j] == origin:
                        continue
                    probs.append(float(p[j]))
        return np.array(probs, dtype=np.float64)
