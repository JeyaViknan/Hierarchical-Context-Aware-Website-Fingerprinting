"""
Stage 2 Training – Session-Level Transformer + Multi-task Head.

Stage 1 encoder is FROZEN.  We train only:
  - Session Transformer (with Transition-Aware Attention)
  - Multi-task Classification Head (site + intent)

The training procedure:
  1. Generate trace embeddings using the frozen Stage 1 encoder
  2. Group embeddings into sessions
  3. Train the Session Transformer + MultitaskHead with combined loss
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hcwf.models.packet_transformer import PacketTransformer
from hcwf.models.session_transformer import SessionTransformer
from hcwf.models.multitask_head import MultitaskHead
from hcwf.training.loss import MultitaskLoss
from hcwf.utils.config import HCWFConfig
from hcwf.utils.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_embeddings(
    encoder: PacketTransformer,
    X: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Generate trace embeddings using the frozen Stage 1 encoder.

    Parameters
    ----------
    encoder    : trained PacketTransformer
    X          : (N, max_trace_len, n_features)
    device     : device string
    batch_size : inference batch size

    Returns
    -------
    embeddings : (N, embedding_dim)
    """
    encoder = encoder.to(device)
    encoder.eval()

    all_embeddings = []
    N = X.shape[0]

    for i in range(0, N, batch_size):
        batch = X[i : i + batch_size].to(device)
        emb = encoder.encode(batch)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def make_session_dataloader(
    embeddings: torch.Tensor,
    site_labels: torch.Tensor,
    intent_labels: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for session-level training."""
    ds = TensorDataset(
        embeddings.float(),
        site_labels.long(),
        intent_labels.long(),
        mask.bool(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_stage2(
    session_model: SessionTransformer,
    multitask_head: MultitaskHead,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    cfg: Optional[HCWFConfig] = None,
    device: str = "cpu",
) -> Dict[str, list]:
    """
    Train Stage 2: Session Transformer + Multi-task Head.

    Parameters
    ----------
    session_model  : SessionTransformer instance
    multitask_head : MultitaskHead instance
    train_loader   : DataLoader yielding (embeddings, site_labels, intent_labels, mask)
    val_loader     : optional validation DataLoader
    cfg            : HCWFConfig
    device         : device string

    Returns
    -------
    history : dict with training metrics per epoch
    """
    if cfg is None:
        cfg = HCWFConfig()

    tcfg = cfg.training
    device = torch.device(device)
    session_model = session_model.to(device)
    multitask_head = multitask_head.to(device)

    # Combined parameters
    params = list(session_model.parameters()) + list(multitask_head.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=tcfg.stage2_lr,
        weight_decay=tcfg.stage2_weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg.stage2_epochs, eta_min=1e-6
    )

    criterion = MultitaskLoss(
        intent_loss_weight=tcfg.intent_loss_weight,
        label_smoothing=tcfg.label_smoothing,
    )

    ckpt_dir = Path(tcfg.checkpoint_dir) / "stage2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, list] = {
        "train_loss": [],
        "train_site_loss": [],
        "train_intent_loss": [],
        "val_loss": [],
        "val_site_accuracy": [],
        "val_intent_accuracy": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(
        f"Stage 2 Training: {tcfg.stage2_epochs} epochs, "
        f"lr={tcfg.stage2_lr}, λ_intent={tcfg.intent_loss_weight}, device={device}"
    )

    for epoch in range(1, tcfg.stage2_epochs + 1):
        t0 = time.time()

        # ---- Train ----
        session_model.train()
        multitask_head.train()
        epoch_loss = 0.0
        epoch_site_loss = 0.0
        epoch_intent_loss = 0.0
        n_batches = 0

        for batch_idx, (emb_batch, site_batch, intent_batch, mask_batch) in enumerate(train_loader):
            emb_batch = emb_batch.to(device)
            site_batch = site_batch.to(device)
            intent_batch = intent_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()

            # Forward: session transformer → multitask head
            h_session, _ = session_model(emb_batch, mask=mask_batch)
            predictions = multitask_head(h_session)

            # For site loss, use the majority label in each session
            # (session-level site classification)
            session_site_labels = _get_session_site_labels(site_batch, mask_batch)

            losses = criterion(
                site_logits=predictions["site_logits"],
                site_labels=session_site_labels,
                intent_logits=predictions["intent_logits"],
                intent_labels=intent_batch,
            )

            losses["total"].backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += losses["total"].item()
            epoch_site_loss += losses["site"].item()
            epoch_intent_loss += losses["intent"].item()
            n_batches += 1

            if batch_idx % tcfg.log_interval == 0 and batch_idx > 0:
                logger.debug(
                    f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Total: {losses['total'].item():.4f} | "
                    f"Site: {losses['site'].item():.4f} | "
                    f"Intent: {losses['intent'].item():.4f}"
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)
        history["train_site_loss"].append(epoch_site_loss / max(n_batches, 1))
        history["train_intent_loss"].append(epoch_intent_loss / max(n_batches, 1))
        history["lr"].append(scheduler.get_last_lr()[0])

        # ---- Validate ----
        val_loss, val_site_acc, val_intent_acc = 0.0, 0.0, 0.0
        if val_loader is not None:
            val_loss, val_site_acc, val_intent_acc = _evaluate_stage2(
                session_model, multitask_head, val_loader, criterion, device
            )

        history["val_loss"].append(val_loss)
        history["val_site_accuracy"].append(val_site_acc)
        history["val_intent_accuracy"].append(val_intent_acc)

        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{tcfg.stage2_epochs} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Site Acc: {val_site_acc:.4f} | "
            f"Intent Acc: {val_intent_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ---- Checkpointing ----
        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _save_stage2_checkpoint(
                session_model, multitask_head, optimizer, epoch,
                ckpt_dir / "best.pt"
            )
            logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            _save_stage2_checkpoint(
                session_model, multitask_head, optimizer, epoch,
                ckpt_dir / f"epoch_{epoch}.pt"
            )

        if (
            val_loader is not None
            and patience_counter >= tcfg.early_stopping_patience
        ):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    _save_stage2_checkpoint(
        session_model, multitask_head, optimizer, epoch,
        ckpt_dir / "final.pt"
    )
    logger.info("Stage 2 training complete.")

    return history


def _get_session_site_labels(
    site_labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract the dominant (most frequent) site label per session.

    site_labels : (B, max_session_len)
    mask        : (B, max_session_len)

    Returns : (B,) integer labels
    """
    B = site_labels.shape[0]
    result = torch.zeros(B, dtype=torch.long, device=site_labels.device)
    for i in range(B):
        valid = site_labels[i][mask[i]]
        if valid.numel() > 0:
            # Mode (most frequent label in session)
            vals, counts = valid.unique(return_counts=True)
            result[i] = vals[counts.argmax()]
        else:
            result[i] = 0
    return result


def _evaluate_stage2(
    session_model: SessionTransformer,
    multitask_head: MultitaskHead,
    loader: DataLoader,
    criterion: MultitaskLoss,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate Stage 2, returning (loss, site_acc, intent_acc)."""
    session_model.eval()
    multitask_head.eval()
    total_loss = 0.0
    all_site_preds, all_site_labels = [], []
    all_intent_preds, all_intent_labels = [], []

    with torch.no_grad():
        for emb_batch, site_batch, intent_batch, mask_batch in loader:
            emb_batch = emb_batch.to(device)
            site_batch = site_batch.to(device)
            intent_batch = intent_batch.to(device)
            mask_batch = mask_batch.to(device)

            h_session, _ = session_model(emb_batch, mask=mask_batch)
            predictions = multitask_head(h_session)
            session_site_labels = _get_session_site_labels(site_batch, mask_batch)

            losses = criterion(
                predictions["site_logits"], session_site_labels,
                predictions["intent_logits"], intent_batch,
            )
            total_loss += losses["total"].item()

            all_site_preds.append(predictions["site_logits"].argmax(dim=-1).cpu().numpy())
            all_site_labels.append(session_site_labels.cpu().numpy())
            all_intent_preds.append(predictions["intent_logits"].argmax(dim=-1).cpu().numpy())
            all_intent_labels.append(intent_batch.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)

    site_preds = np.concatenate(all_site_preds)
    site_labels = np.concatenate(all_site_labels)
    intent_preds = np.concatenate(all_intent_preds)
    intent_labels = np.concatenate(all_intent_labels)

    site_metrics = compute_classification_metrics(site_labels, site_preds, task_name="site")
    intent_metrics = compute_classification_metrics(intent_labels, intent_preds, task_name="intent")

    return avg_loss, site_metrics["site/accuracy"], intent_metrics["intent/accuracy"]


def _save_stage2_checkpoint(
    session_model: SessionTransformer,
    multitask_head: MultitaskHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "session_model_state_dict": session_model.state_dict(),
            "multitask_head_state_dict": multitask_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_stage2_checkpoint(
    session_model: SessionTransformer,
    multitask_head: MultitaskHead,
    path: str,
    device: str = "cpu",
) -> Tuple[SessionTransformer, MultitaskHead]:
    """Load a Stage 2 checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    session_model.load_state_dict(checkpoint["session_model_state_dict"])
    multitask_head.load_state_dict(checkpoint["multitask_head_state_dict"])
    logger.info(f"Loaded Stage 2 checkpoint from {path} (epoch {checkpoint['epoch']})")
    return session_model, multitask_head
