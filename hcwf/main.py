"""
HC-WF Main Entry Point
=======================

Runs the full Hierarchical Context-Aware Website Fingerprinting pipeline
end-to-end using dummy data (no dataset required).

Usage:
    python main.py                          # Run with defaults
    python main.py --config config.yaml     # Run with custom config
    python main.py --stage 1                # Train Stage 1 only
    python main.py --stage 2                # Train Stage 2 only
    python main.py --demo                   # Quick demo with minimal epochs
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hcwf.utils.config import HCWFConfig
from hcwf.data.preprocessing import generate_dummy_traces
from hcwf.data.session_builder import generate_dummy_sessions
from hcwf.models.packet_transformer import PacketTransformer
from hcwf.models.session_transformer import SessionTransformer
from hcwf.models.multitask_head import MultitaskHead
from hcwf.training.train_stage1 import train_stage1, make_trace_dataloader
from hcwf.training.train_stage2 import (
    train_stage2,
    generate_embeddings,
    make_session_dataloader,
)
from hcwf.inference.predictor import HCWFPredictor
from hcwf.utils.metrics import compute_classification_metrics

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hcwf.main")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HC-WF: Hierarchical Context-Aware Website Fingerprinting"
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML/JSON config file (optional)",
    )
    p.add_argument(
        "--stage", type=int, default=0, choices=[0, 1, 2],
        help="Which stage to train: 0=both (default), 1=Stage1 only, 2=Stage2 only",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Quick demo mode with minimal epochs and small data",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, cuda, mps",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load / create config ─────────────────────────────────────────────
    if args.config:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            cfg = HCWFConfig.from_yaml(args.config)
        else:
            cfg = HCWFConfig.from_json(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        cfg = HCWFConfig()
        logger.info("Using default configuration")

    # Override with CLI args
    cfg.training.seed = args.seed
    cfg.training.device = args.device

    if args.demo:
        # Shrink everything for a fast, memory-friendly demo
        cfg.training.stage1_epochs = 3
        cfg.training.stage2_epochs = 3
        cfg.training.early_stopping_patience = 100  # disable
        cfg.preprocess.max_trace_len = 500           # much shorter traces
        cfg.packet_transformer.d_model = 64
        cfg.packet_transformer.nhead = 4
        cfg.packet_transformer.num_layers = 2
        cfg.packet_transformer.dim_feedforward = 128
        cfg.packet_transformer.embedding_dim = 64
        cfg.session_transformer.d_model = 64
        cfg.session_transformer.nhead = 4
        cfg.session_transformer.num_layers = 1
        cfg.session_transformer.dim_feedforward = 128
        cfg.multitask.n_sites = 20
        cfg.multitask.n_intents = 4
        cfg.multitask.hidden_dim = 64
        n_traces = 200
        n_sessions = 80
        logger.info("🚀 Demo mode: using minimal epochs and small data")
    else:
        n_traces = 1000
        n_sessions = 300

    device = cfg.training.resolve_device()
    logger.info(f"Device: {device}")

    # Set seeds
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # ── Save config ──────────────────────────────────────────────────────
    config_dir = Path(cfg.training.checkpoint_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(str(config_dir / "config.yaml"))
    logger.info(f"Config saved to {config_dir / 'config.yaml'}")

    n_sites = cfg.multitask.n_sites
    n_intents = cfg.multitask.n_intents

    # ════════════════════════════════════════════════════════════════════
    # STAGE 1: Packet-Level Transformer
    # ════════════════════════════════════════════════════════════════════

    if args.stage in (0, 1):
        logger.info("=" * 70)
        logger.info("STAGE 1: Packet-Level Transformer (Trace Encoder)")
        logger.info("=" * 70)

        # Generate dummy trace data
        logger.info(f"Generating {n_traces} dummy traces for {n_sites} sites...")
        X, y = generate_dummy_traces(
            n_traces=n_traces,
            n_sites=n_sites,
            cfg=cfg.preprocess,
            seed=cfg.training.seed,
        )
        logger.info(f"Trace data shape: X={tuple(X.shape)}, y={tuple(y.shape)}")

        # Train / val split
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_loader = make_trace_dataloader(X_train, y_train, batch_size=cfg.training.stage1_batch_size)
        val_loader = make_trace_dataloader(X_val, y_val, batch_size=cfg.training.stage1_batch_size, shuffle=False)

        # Create model
        packet_model = PacketTransformer(
            n_features=cfg.preprocess.n_features,
            n_classes=n_sites,
            cfg=cfg.packet_transformer,
        )
        logger.info(
            f"PacketTransformer: "
            f"{sum(p.numel() for p in packet_model.parameters()):,} parameters"
        )

        # Train
        t0 = time.time()
        history1 = train_stage1(
            model=packet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
        )
        logger.info(f"Stage 1 completed in {time.time() - t0:.1f}s")
        logger.info(
            f"Final metrics → "
            f"Train Loss: {history1['train_loss'][-1]:.4f}, "
            f"Val Loss: {history1['val_loss'][-1]:.4f}, "
            f"Val Accuracy: {history1['val_accuracy'][-1]:.4f}"
        )
    else:
        # Load existing Stage 1 model
        packet_model = PacketTransformer(
            n_features=cfg.preprocess.n_features,
            n_classes=n_sites,
            cfg=cfg.packet_transformer,
        )
        stage1_ckpt = Path(cfg.training.checkpoint_dir) / "stage1" / "final.pt"
        if stage1_ckpt.exists():
            from hcwf.training.train_stage1 import load_stage1_checkpoint
            packet_model = load_stage1_checkpoint(packet_model, str(stage1_ckpt), device)
        else:
            logger.warning("No Stage 1 checkpoint found; using untrained encoder")

    # ════════════════════════════════════════════════════════════════════
    # STAGE 2: Session-Level Transformer
    # ════════════════════════════════════════════════════════════════════

    if args.stage in (0, 2):
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: Session-Level Transformer (Context Encoder)")
        logger.info("=" * 70)

        # Generate dummy session data
        logger.info(f"Generating {n_sessions} dummy sessions...")
        emb_dim = cfg.packet_transformer.embedding_dim
        max_sess_len = cfg.session.max_session_len

        session_emb, session_site, session_intent, session_mask = generate_dummy_sessions(
            n_sessions=n_sessions,
            n_sites=n_sites,
            n_intents=n_intents,
            embedding_dim=emb_dim,
            max_session_len=max_sess_len,
            seed=cfg.training.seed,
        )
        logger.info(
            f"Session data shape: embeddings={tuple(session_emb.shape)}, "
            f"site_labels={tuple(session_site.shape)}, "
            f"intent_labels={tuple(session_intent.shape)}"
        )

        # Train / val split
        split = int(0.8 * n_sessions)
        train_loader2 = make_session_dataloader(
            session_emb[:split], session_site[:split],
            session_intent[:split], session_mask[:split],
            batch_size=cfg.training.stage2_batch_size,
        )
        val_loader2 = make_session_dataloader(
            session_emb[split:], session_site[split:],
            session_intent[split:], session_mask[split:],
            batch_size=cfg.training.stage2_batch_size,
            shuffle=False,
        )

        # Create models
        session_model = SessionTransformer(
            cfg=cfg.session_transformer,
            transition_cfg=cfg.transition_attention,
        )
        multitask_head = MultitaskHead(
            cfg=cfg.multitask,
            input_dim=cfg.session_transformer.d_model,
        )
        logger.info(
            f"SessionTransformer: "
            f"{sum(p.numel() for p in session_model.parameters()):,} parameters"
        )
        logger.info(
            f"MultitaskHead: "
            f"{sum(p.numel() for p in multitask_head.parameters()):,} parameters"
        )

        # Freeze Stage 1
        for param in packet_model.parameters():
            param.requires_grad = False
        logger.info("Stage 1 encoder frozen ❄️")

        # Train
        t0 = time.time()
        history2 = train_stage2(
            session_model=session_model,
            multitask_head=multitask_head,
            train_loader=train_loader2,
            val_loader=val_loader2,
            cfg=cfg,
            device=device,
        )
        logger.info(f"Stage 2 completed in {time.time() - t0:.1f}s")
        logger.info(
            f"Final metrics → "
            f"Loss: {history2['train_loss'][-1]:.4f}, "
            f"Site Acc: {history2['val_site_accuracy'][-1]:.4f}, "
            f"Intent Acc: {history2['val_intent_accuracy'][-1]:.4f}"
        )

    # ════════════════════════════════════════════════════════════════════
    # DEMO INFERENCE
    # ════════════════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 70)
    logger.info("DEMO INFERENCE")
    logger.info("=" * 70)

    predictor = HCWFPredictor(cfg, device=device)

    # Generate a small demo session
    rng = np.random.RandomState(cfg.training.seed)
    demo_traces = []
    for _ in range(3):
        n_pkt = rng.randint(100, 500)
        dirs = rng.choice([1, -1], size=n_pkt)
        mags = rng.exponential(800, size=n_pkt)
        demo_traces.append((dirs * mags).astype(np.float32))

    result = predictor.predict_session(demo_traces, return_attention=True)

    logger.info(f"Session website prediction: {result['site_prediction']}")
    logger.info(f"Session intent prediction:  {result['intent_prediction']}")
    logger.info(f"Per-trace predictions:      {result['per_trace_sites']}")

    # Model summary
    summary = predictor.get_model_summary()
    logger.info(f"\nModel parameter counts:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v:,}")

    logger.info("\n✅ HC-WF pipeline completed successfully!")
    logger.info(
        f"Run 'streamlit run hcwf/app/streamlit_app.py' to launch the interactive UI."
    )


if __name__ == "__main__":
    main()
