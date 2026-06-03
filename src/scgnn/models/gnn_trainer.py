"""
GNN training bridge — the piece that was missing (run_ladder previously skipped GNNs).

Trains GraphSAGE / GAT as a per-node binary classifier over hourly directed
snapshots, masked to active non-origin nodes, with class-weighted BCE and early
stopping.  ``predict_episodes`` returns probabilities in the SAME row order as
``dataset.tabular_from_episodes`` so the GNN is scored on identical samples to the
tabular ladder.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from scgnn.data.dataset import load_episode, to_pyg_snapshots
from scgnn.models.gnn import GATContagion, GraphSAGEContagion


def _build_model(kind: str, in_dim: int, hidden: int = 64, layers: int = 3,
                 dropout: float = 0.2, edge_dim: int = 4) -> torch.nn.Module:
    if kind == "graphsage":
        return GraphSAGEContagion(in_dim, hidden, layers, dropout)
    if kind == "gat":
        return GATContagion(in_dim, hidden, heads=4, num_layers=layers,
                            dropout=dropout, edge_dim=edge_dim)
    raise ValueError(kind)


class GNNContagionTrainer:
    def __init__(self, kind: str = "graphsage", horizon: int = 60, hidden: int = 64,
                 layers: int = 3, dropout: float = 0.2, lr: float = 1e-3,
                 epochs: int = 100, patience: int = 12, seed: int = 42,
                 device: str = "cpu"):
        self.kind = kind; self.horizon = horizon; self.hidden = hidden
        self.layers = layers; self.dropout = dropout; self.lr = lr
        self.epochs = epochs; self.patience = patience; self.seed = seed
        self.device = device; self.model: Optional[torch.nn.Module] = None
        self.in_dim: Optional[int] = None

    def _episode_data(self, names: List[str]) -> List:
        data = []
        for name in names:
            b = load_episode(name)
            for _, d in to_pyg_snapshots(b, self.horizon):
                if d.eval_mask.sum() == 0:
                    continue
                data.append(d.to(self.device))
        return data

    def fit(self, train_names: List[str], val_names: Optional[List[str]] = None) -> "GNNContagionTrainer":
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        train_data = self._episode_data(train_names)
        if not train_data:
            raise RuntimeError("no trainable snapshots")
        self.in_dim = train_data[0].x.shape[1]
        self.model = _build_model(self.kind, self.in_dim, self.hidden, self.layers,
                                  self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        # class weight from train labels (positive is rare)
        pos = sum(float(d.y[d.eval_mask].sum()) for d in train_data)
        tot = sum(int(d.eval_mask.sum()) for d in train_data)
        pos_weight = torch.tensor([(tot - pos) / max(pos, 1.0)], device=self.device)

        val_data = self._episode_data(val_names) if val_names else None
        best_val, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            self.model.train()
            np.random.shuffle(train_data)
            for d in train_data:
                opt.zero_grad()
                logits = self.model(d.x, d.edge_index, d.edge_attr).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits[d.eval_mask], d.y[d.eval_mask], pos_weight=pos_weight)
                loss.backward(); opt.step()
            # early stopping
            ref = val_data if val_data else train_data
            vloss = self._epoch_loss(ref, pos_weight)
            if vloss < best_val - 1e-4:
                best_val, best_state, bad = vloss, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _epoch_loss(self, data, pos_weight) -> float:
        self.model.eval(); tot, n = 0.0, 0
        for d in data:
            logits = self.model(d.x, d.edge_index, d.edge_attr).squeeze(-1)
            tot += float(F.binary_cross_entropy_with_logits(
                logits[d.eval_mask], d.y[d.eval_mask], pos_weight=pos_weight)) * int(d.eval_mask.sum())
            n += int(d.eval_mask.sum())
        return tot / max(n, 1)

    @torch.no_grad()
    def predict_episodes(self, names: List[str]) -> np.ndarray:
        """Probabilities in tabular_from_episodes row order (episode, snapshot, active node)."""
        self.model.eval()
        probs: List[float] = []
        for name in names:
            b = load_episode(name)
            node_strs = b["node_strs"]; origin = b["origin"]; active = b["active"]
            for si, d in to_pyg_snapshots(b, self.horizon):
                logits = self.model(d.x, d.edge_index, d.edge_attr).squeeze(-1)
                p = torch.sigmoid(logits).cpu().numpy()
                for j in range(len(node_strs)):
                    if not active[si, j] or node_strs[j] == origin:
                        continue
                    probs.append(float(p[j]))
        return np.array(probs, dtype=np.float64)
