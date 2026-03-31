"""
Stage 2 – Session-Level Transformer (Context Encoder).

Operates on a sequence of trace embeddings (z_1, z_2, ..., z_L)
produced by Stage 1, and models inter-trace context using
Transition-Aware Attention.

Architecture
------------
  [z_1, z_2, ..., z_L] → Positional Encoding → N × TransitionAwareEncoderBlock → Mean Pool → h_session

Each encoder block contains:
  1. Transition-Aware Multi-Head Attention (with T_ij bias)
  2. Layer Norm + Residual Connection
  3. Feed-forward Network
  4. Layer Norm + Residual Connection
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from hcwf.utils.config import SessionTransformerConfig, TransitionAttentionConfig
from hcwf.models.transition_attention import TransitionAwareAttention


class TransitionAwareEncoderBlock(nn.Module):
    """
    A single encoder block using Transition-Aware Attention
    instead of standard multi-head attention.

    Pre-norm architecture for training stability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        max_session_len: int,
        dropout: float = 0.1,
        learnable_bias: bool = True,
    ):
        super().__init__()

        # Transition-Aware Attention
        self.attn = TransitionAwareAttention(
            d_model=d_model,
            nhead=nhead,
            max_session_len=max_session_len,
            dropout=dropout,
            learnable_bias=learnable_bias,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        x    : (batch, session_len, d_model)
        mask : optional (batch, session_len), True = valid

        Returns
        -------
        output       : (batch, session_len, d_model)
        attn_weights : optional (batch, nhead, session_len, session_len)
        """
        # Pre-norm attention with residual
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, mask=mask, return_attention=return_attention)
        x = residual + attn_out

        # Pre-norm FFN with residual
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x, attn_weights


class SessionPositionalEncoding(nn.Module):
    """
    Learnable positional encoding for session positions.

    Since sessions are short (2-5 traces), we use simple learnable
    position embeddings rather than sinusoidal encoding.
    """

    def __init__(self, d_model: int, max_len: int = 10, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device, dtype=torch.long)
        return self.dropout(x + self.pos_embedding(positions).unsqueeze(0))


class SessionTransformer(nn.Module):
    """
    Session-Level Transformer with Transition-Aware Attention.

    Takes a sequence of trace embeddings from Stage 1 and produces
    a session-level representation h_session that captures browsing
    context and inter-trace relationships.

    Forward signature::

        forward(embeddings, mask=None) -> h_session

    Input:  (batch, max_session_len, embedding_dim)
    Output: (batch, d_model) — the session representation h_session
    """

    def __init__(
        self,
        cfg: SessionTransformerConfig,
        transition_cfg: TransitionAttentionConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # Input projection (embedding_dim -> d_model, they may differ)
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # Positional encoding for session positions
        self.pos_encoder = SessionPositionalEncoding(
            cfg.d_model,
            max_len=cfg.max_session_len,
            dropout=cfg.dropout,
        )

        # Stack of Transition-Aware Encoder Blocks
        self.layers = nn.ModuleList([
            TransitionAwareEncoderBlock(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                max_session_len=transition_cfg.max_session_len,
                dropout=cfg.dropout,
                learnable_bias=transition_cfg.learnable_bias,
            )
            for _ in range(cfg.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        embeddings: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """
        Parameters
        ----------
        embeddings       : (batch, session_len, embedding_dim)
        mask             : optional (batch, session_len), True = valid
        return_attention : if True, collect attention weights from all layers

        Returns
        -------
        h_session    : (batch, d_model) — mean-pooled session representation
        attn_weights : list of attention weight tensors (one per layer)
        """
        h = self.input_proj(embeddings)
        h = self.pos_encoder(h)

        all_attn_weights = []
        for layer in self.layers:
            h, attn_w = layer(h, mask=mask, return_attention=return_attention)
            all_attn_weights.append(attn_w)

        h = self.final_norm(h)

        # Mean pooling over session (respecting mask)
        if mask is not None:
            valid = mask.unsqueeze(-1).float()  # (B, L, 1)
            h_session = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            h_session = h.mean(dim=1)  # (B, d_model)

        return h_session, all_attn_weights

    def get_all_transition_biases(self) -> List[Tensor]:
        """Return transition bias matrices from all layers for visualisation."""
        biases = []
        for layer in self.layers:
            biases.append(layer.attn.get_transition_bias())
        return biases
