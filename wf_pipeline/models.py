from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TransformerConfig:
    """
    WF-Transformer style per-trace classifier.

    d_model: embedding dimension for packet-level features
    nhead: number of attention heads
    num_layers: number of encoder layers
    dim_ff: feed-forward inner dimension
    dropout: transformer dropout
    """

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 128
    dropout: float = 0.1


class TraceTransformer(nn.Module):
    """
    Simple Transformer encoder over packet-level features.

    Input: (batch, seq_len, n_features) where features can include
    packet direction, (normalized) size, and inter-arrival time.
    Output: (batch, n_classes) logits over website categories.
    """

    def __init__(self, n_features: int, n_classes: int, cfg: TransformerConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.cls_head = nn.Linear(cfg.d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        h = self.input_proj(x)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.cls_head(pooled)


@dataclass(frozen=True)
class ContextRNNConfig:
    """
    BiLSTM context model over sequences of website predictions.

    hidden_size: LSTM hidden dimension per direction.
    """

    hidden_size: int = 64


class ContextBiLSTM(nn.Module):
    """
    Sequence model that maps a sequence of website categories to a
    session-level intent class.
    """

    def __init__(self, n_sites: int, n_intents: int, cfg: ContextRNNConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_sites, cfg.hidden_size)
        self.lstm = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * cfg.hidden_size, n_intents)

    def forward(self, seq_sites: torch.Tensor) -> torch.Tensor:
        # seq_sites: (B, T) of integer site indices
        emb = self.embed(seq_sites)
        out, _ = self.lstm(emb)
        pooled = out.mean(dim=1)
        return self.fc(pooled)


def train_simple_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 3,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> None:
    """
    Lightweight training loop for both the trace Transformer and
    the context BiLSTM models. This is intentionally minimal and
    designed for interactive demos, not large-scale training.
    """

    device = torch.device(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(num_epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()


def make_trace_dataloader(
    X_seq: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True
) -> DataLoader:
    X_t = torch.from_numpy(X_seq).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_context_dataloader(
    seq_indices: np.ndarray, intent_labels: np.ndarray, batch_size: int = 64, shuffle: bool = True
) -> DataLoader:
    xs = torch.from_numpy(seq_indices).long()
    ys = torch.from_numpy(intent_labels).long()
    ds = TensorDataset(xs, ys)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

