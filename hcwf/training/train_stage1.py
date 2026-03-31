"""
Stage 1 Training – Packet-Level Transformer (Trace Encoder).

Trains the packet transformer independently on per-trace website
classification.  After training, the encoder weights are saved so
that Stage 2 can use the frozen encoder to produce trace embeddings.

Features:
  - AdamW optimiser with weight decay
  - Cosine-annealing learning rate schedule
  - Gradient clipping
  - Checkpoint saving (best + periodic)
  - Training logging
  - Early stopping
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hcwf.models.packet_transformer import PacketTransformer
from hcwf.training.loss import Stage1Loss
from hcwf.utils.config import HCWFConfig
from hcwf.utils.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


def make_trace_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from trace tensors and labels."""
    ds = TensorDataset(X.float(), y.long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_stage1(
    model: PacketTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    cfg: Optional[HCWFConfig] = None,
    device: str = "cpu",
) -> Dict[str, list]:
    """
    Train the Stage 1 Packet-Level Transformer.

    Parameters
    ----------
    model        : PacketTransformer instance
    train_loader : DataLoader yielding (X_batch, y_batch)
    val_loader   : optional validation DataLoader
    cfg          : HCWFConfig (uses training sub-config)
    device       : device string

    Returns
    -------
    history : dict with keys "train_loss", "val_loss", "val_accuracy"
    """
    if cfg is None:
        cfg = HCWFConfig()

    tcfg = cfg.training
    device = torch.device(device)
    model = model.to(device)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.stage1_lr,
        weight_decay=tcfg.stage1_weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg.stage1_epochs, eta_min=1e-6
    )

    # Loss
    criterion = Stage1Loss(label_smoothing=tcfg.label_smoothing)

    # Checkpoint directory
    ckpt_dir = Path(tcfg.checkpoint_dir) / "stage1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(
        f"Stage 1 Training: {tcfg.stage1_epochs} epochs, "
        f"lr={tcfg.stage1_lr}, device={device}"
    )

    for epoch in range(1, tcfg.stage1_epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx % tcfg.log_interval == 0 and batch_idx > 0:
                logger.debug(
                    f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)
        history["lr"].append(scheduler.get_last_lr()[0])

        # ---- Validate ----
        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            val_loss, val_acc = _evaluate_stage1(model, val_loader, criterion, device)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{tcfg.stage1_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ---- Checkpointing ----
        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _save_checkpoint(model, optimizer, epoch, ckpt_dir / "best.pt")
            logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % 5 == 0:
            _save_checkpoint(model, optimizer, epoch, ckpt_dir / f"epoch_{epoch}.pt")

        # ---- Early stopping ----
        if (
            val_loader is not None
            and patience_counter >= tcfg.early_stopping_patience
        ):
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {tcfg.early_stopping_patience} epochs)"
            )
            break

    # Save final model
    _save_checkpoint(model, optimizer, epoch, ckpt_dir / "final.pt")
    logger.info("Stage 1 training complete.")

    return history


def _evaluate_stage1(
    model: PacketTransformer,
    loader: DataLoader,
    criterion: Stage1Loss,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate on validation set, returning (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_classification_metrics(y_true, y_pred, task_name="site")
    return avg_loss, metrics["site/accuracy"]


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    """Save model and optimizer state."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_stage1_checkpoint(
    model: PacketTransformer,
    path: str,
    device: str = "cpu",
) -> PacketTransformer:
    """Load a Stage 1 checkpoint into the model."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded Stage 1 checkpoint from {path} (epoch {checkpoint['epoch']})")
    return model
