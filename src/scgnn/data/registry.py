"""
Canonical NodeID schema with stable integer IDs.

The registry is the single source of truth for node ordering across all
graph snapshots, feature matrices, and model inputs.  IDs are stable across
runs as long as the node universe in experiment.yaml does not change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, order=True)
class NodeID:
    asset: str
    venue: str
    fee_tier: str = ""

    def __str__(self) -> str:
        parts = [self.asset, self.venue]
        if self.fee_tier:
            parts.append(self.fee_tier)
        return "/".join(parts)

    @classmethod
    def from_str(cls, s: str) -> "NodeID":
        parts = s.split("/")
        if len(parts) == 3:
            return cls(parts[0], parts[1], parts[2])
        if len(parts) == 2:
            return cls(parts[0], parts[1])
        raise ValueError(f"Cannot parse NodeID from '{s}'")


class NodeRegistry:
    """Bidirectional map between NodeID and stable integer index."""

    def __init__(self, nodes: List[NodeID]):
        # Sorted for deterministic ordering
        self._nodes: List[NodeID] = sorted(set(nodes))
        self._id_to_idx: Dict[NodeID, int] = {n: i for i, n in enumerate(self._nodes)}

    @classmethod
    def from_config(
        cls,
        assets: List[str],
        venues: List[str],
        fee_tiers: Optional[Dict[str, List[str]]] = None,
    ) -> "NodeRegistry":
        nodes = []
        for asset in sorted(assets):
            for venue in sorted(venues):
                tiers = fee_tiers.get(venue, [""]) if fee_tiers else [""]
                for tier in tiers:
                    nodes.append(NodeID(asset, venue, tier))
        return cls(nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def idx(self, node: NodeID) -> int:
        return self._id_to_idx[node]

    def node(self, idx: int) -> NodeID:
        return self._nodes[idx]

    def node_strs(self) -> List[str]:
        return [str(n) for n in self._nodes]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [{"idx": i, "node": str(n)} for i, n in enumerate(self._nodes)]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "NodeRegistry":
        with open(path) as f:
            records = json.load(f)
        nodes = [NodeID.from_str(r["node"]) for r in sorted(records, key=lambda r: r["idx"])]
        return cls(nodes)
