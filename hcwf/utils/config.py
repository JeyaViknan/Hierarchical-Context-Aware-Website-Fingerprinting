"""
Centralized configuration for the HC-WF pipeline.

All hyperparameters are gathered here so that the entire system can be
configured from a single YAML / dict / CLI without touching source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import json
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Data pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Controls how raw packet traces are converted to tensors."""

    max_trace_len: int = 2000          # Pad / truncate each trace to this length
    clip_size: int = 20000             # Clip signed sizes to [-clip, clip]
    include_timing: bool = True        # Whether to include IAT features
    n_features: int = 2                # Number of per-packet features (direction + IAT)


@dataclass
class SessionConfig:
    """Controls how individual traces are grouped into sessions."""

    min_session_len: int = 2
    max_session_len: int = 5
    time_gap_threshold: float = 30.0   # seconds – traces > gap apart start new session
    simulate_timestamps: bool = True   # Generate synthetic timestamps when unavailable


# ---------------------------------------------------------------------------
# Model architecture config
# ---------------------------------------------------------------------------

@dataclass
class PacketTransformerConfig:
    """Stage-1 Packet-Level Transformer (Trace Encoder)."""

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    embedding_dim: int = 128          # Dimension of the output embedding z_j
    use_relative_pos: bool = True     # Use relative positional encoding


@dataclass
class SessionTransformerConfig:
    """Stage-2 Session-Level Transformer (Context Encoder)."""

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_session_len: int = 5          # Maximum number of traces per session


@dataclass
class TransitionAttentionConfig:
    """Transition-Aware Attention parameters."""

    max_session_len: int = 5
    learnable_bias: bool = True       # Use learnable T_ij matrix


@dataclass
class MultitaskConfig:
    """Multi-task classification head."""

    n_sites: int = 100                # Number of website classes
    n_intents: int = 6                # Number of intent categories
    hidden_dim: int = 128
    dropout: float = 0.2


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for training both stages."""

    # Stage 1
    stage1_epochs: int = 30
    stage1_lr: float = 1e-3
    stage1_batch_size: int = 64
    stage1_weight_decay: float = 1e-4

    # Stage 2
    stage2_epochs: int = 20
    stage2_lr: float = 5e-4
    stage2_batch_size: int = 32
    stage2_weight_decay: float = 1e-4

    # Loss
    intent_loss_weight: float = 0.3   # λ in L = L_site + λ * L_intent
    label_smoothing: float = 0.1

    # General
    seed: int = 42
    device: str = "auto"              # "auto", "cpu", "cuda", "mps"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10            # Print every N batches
    early_stopping_patience: int = 5

    def resolve_device(self) -> str:
        """Auto-detect the best available device."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ---------------------------------------------------------------------------
# Top-level system config
# ---------------------------------------------------------------------------

@dataclass
class HCWFConfig:
    """Master configuration aggregating all sub-configs."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    packet_transformer: PacketTransformerConfig = field(default_factory=PacketTransformerConfig)
    session_transformer: SessionTransformerConfig = field(default_factory=SessionTransformerConfig)
    transition_attention: TransitionAttentionConfig = field(default_factory=TransitionAttentionConfig)
    multitask: MultitaskConfig = field(default_factory=MultitaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(s)
        return s

    def to_yaml(self, path: Optional[str] = None) -> str:
        s = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            Path(path).write_text(s)
        return s

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HCWFConfig":
        return cls(
            preprocess=PreprocessConfig(**d.get("preprocess", {})),
            session=SessionConfig(**d.get("session", {})),
            packet_transformer=PacketTransformerConfig(**d.get("packet_transformer", {})),
            session_transformer=SessionTransformerConfig(**d.get("session_transformer", {})),
            transition_attention=TransitionAttentionConfig(**d.get("transition_attention", {})),
            multitask=MultitaskConfig(**d.get("multitask", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "HCWFConfig":
        d = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(d)

    @classmethod
    def from_json(cls, path: str) -> "HCWFConfig":
        d = json.loads(Path(path).read_text())
        return cls.from_dict(d)
