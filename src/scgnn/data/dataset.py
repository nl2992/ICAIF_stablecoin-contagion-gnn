"""
Loaders for the real dataset produced by ``scgnn.data.realbuild``.

Two views of the SAME samples (one per active, non-origin (snapshot, node)):
- tabular:  X (M, F), y (M,), meta DataFrame  -> classical / sequence models
- graph:    per-episode dict of snapshots       -> GNN models

``EpisodeGraph`` exposes, for a built episode, the list of PyG snapshots and a
helper to gather per-(snapshot, node) predictions back into tabular order so the
GNN is scored on exactly the rows the tabular models see.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROC = Path("data/processed")


def load_feature_names(proc: Path = PROC) -> List[str]:
    return json.loads((proc / "feature_names.json").read_text())


def load_manifest(proc: Path = PROC) -> dict:
    return json.loads((proc / "dataset_manifest.json").read_text())


def load_tabular(split: str, horizon: int, proc: Path = PROC
                 ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X = np.load(proc / "tabular" / f"X_{split}.npy")
    y = np.load(proc / "tabular" / f"y_{split}_h{horizon}.npy")
    meta = pd.read_parquet(proc / "tabular" / f"meta_{split}.parquet")
    return X, y, meta


def load_episode(name: str, proc: Path = PROC) -> dict:
    with open(proc / "graphs" / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def list_episodes(proc: Path = PROC) -> List[str]:
    return [p.stem for p in sorted((proc / "graphs").glob("*.pkl"))]


def episode_split_map(proc: Path = PROC) -> Dict[str, str]:
    return {n: load_episode(n, proc)["split"] for n in list_episodes(proc)}


def tabular_from_episodes(names: List[str], horizon: int, feat_names: List[str],
                          proc: Path = PROC) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Reconstruct tabular (X, y, meta) for an arbitrary set of episodes (for LOEO)."""
    Xs, ys, rows = [], [], []
    for name in names:
        b = load_episode(name, proc)
        S, N = b["active"].shape
        node_strs = b["node_strs"]
        for si in range(S):
            for j in range(N):
                if not b["active"][si, j] or node_strs[j] == b["origin"]:
                    continue
                Xs.append(b["X"][si, j])
                ys.append(int(b["labels"][horizon][si, j]))
                rows.append({"episode": name, "t": b["hourly_idx"][si],
                             "node": node_strs[j], "snapshot": si, "node_idx": j})
    if not Xs:
        F = len(feat_names)
        return np.empty((0, F), np.float32), np.empty((0,), np.int8), pd.DataFrame()
    return (np.stack(Xs).astype(np.float32),
            np.array(ys, dtype=np.int8),
            pd.DataFrame(rows))


def to_pyg_snapshots(b: dict, horizon: int):
    """Yield (snapshot_idx, Data) for a built episode, with per-node label + active mask."""
    import os
    import torch
    from torch_geometric.data import Data
    X = b["X"]; active = b["active"]; y = b["labels"][horizon]
    origin = b["origin"]; node_strs = b["node_strs"]
    origin_idx = node_strs.index(origin) if origin in node_strs else -1
    # Degree-preserving edge-rewiring null (referee test): when SCGNN_REWIRE_SEED is
    # set, permute edge destinations to destroy the specific directed lead-lag
    # topology while preserving edge count, source out-degrees, and edge attributes.
    _rewire_seed = os.environ.get("SCGNN_REWIRE_SEED")
    for si, snap in enumerate(b["snapshots"]):
        x = torch.tensor(X[si], dtype=torch.float32)
        ei = torch.tensor(snap["edge_index"], dtype=torch.long)
        ea = torch.tensor(snap["edge_attr"], dtype=torch.float32)
        if _rewire_seed is not None and ei.shape[1] > 1:
            g = torch.Generator().manual_seed(int(_rewire_seed) + si + int(ei.shape[1]))
            ei = ei.clone()
            ei[1] = ei[1][torch.randperm(ei.shape[1], generator=g)]
        yt = torch.tensor(y[si], dtype=torch.float32)
        mask = torch.tensor(active[si], dtype=torch.bool)
        if origin_idx >= 0:
            mask[origin_idx] = False  # exclude origin from train/eval
        d = Data(x=x, edge_index=ei, edge_attr=ea, y=yt)
        d.eval_mask = mask
        d.snapshot = si
        yield si, d
