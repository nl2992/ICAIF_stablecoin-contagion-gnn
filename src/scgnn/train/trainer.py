"""Shared training loop for PyTorch models (LSTM, GraphSAGE, GAT)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scgnn.features.labels import weighted_f1_score


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
        loss = criterion(logits.squeeze(-1), batch.y.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
        preds.append((logits.squeeze(-1).sigmoid() > 0.5).cpu().long())
        labels.append(batch.y.cpu().long())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return {"weighted_f1": weighted_f1_score(labels, preds)}


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    pos_weight: Optional[float] = None,
    checkpoint_dir: Optional[Path] = None,
    device: str = "cpu",
) -> dict:
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience // 2, factor=0.5)

    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_f1, best_epoch, wait = 0.0, 0, 0
    history = []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        f1 = metrics["weighted_f1"]
        scheduler.step(-f1)
        history.append({"epoch": epoch, "loss": loss, **metrics})

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            wait = 0
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                break

    return {"best_f1": best_f1, "best_epoch": best_epoch, "history": history}
