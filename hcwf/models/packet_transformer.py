"""
Stage 1 – Packet-Level Transformer (Trace Encoder).

Processes a single traffic trace (sequence of packet features) and
produces a fixed-dimensional embedding vector z_j that captures the
unique fingerprint of the visited website.

Architecture
------------
  Input → Linear projection → Positional Encoding → N × TransformerEncoderLayer → Mean Pool → z_j

Features:
  • Relative positional encoding via learnable embeddings
  • Multi-head self-attention over packet sequences
  • Feed-forward expansion layers
  • Dropout for regularisation
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn, Tensor

from hcwf.utils.config import PacketTransformerConfig


class RelativePositionalEncoding(nn.Module):
    """
    Learnable relative positional encoding.

    Instead of fixed sinusoidal encodings, this module uses learnable
    position embeddings that can capture relative temporal patterns
    in packet sequences.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model) with positional info added
        """
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device, dtype=torch.long)
        pos_emb = self.pos_embedding(positions)  # (L, D)
        return self.dropout(x + pos_emb.unsqueeze(0))


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (non-learnable fallback)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PacketTransformer(nn.Module):
    """
    Packet-Level Transformer Encoder.

    Converts a variable-length packet sequence into a fixed-dimensional
    embedding that captures the website's traffic fingerprint.

    Forward signature::

        forward(trace_sequence: Tensor) -> Tensor

    Input:  (batch, max_trace_len, n_features)
    Output: (batch, embedding_dim) — the trace embedding z_j
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        cfg: PacketTransformerConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        # Project raw packet features to model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

        # Positional encoding
        if cfg.use_relative_pos:
            self.pos_encoder = RelativePositionalEncoding(
                cfg.d_model, dropout=cfg.dropout
            )
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(
                cfg.d_model, dropout=cfg.dropout
            )

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )

        # Embedding projection (d_model -> embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
        )

        # Classification head for Stage 1 pre-training
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embedding_dim, n_classes),
        )

    def encode(
        self,
        trace_sequence: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Produce the embedding vector z_j without the classification head.

        Parameters
        ----------
        trace_sequence : (batch, seq_len, n_features)
        src_key_padding_mask : optional (batch, seq_len) boolean mask,
                               True where tokens are padding

        Returns
        -------
        embedding : (batch, embedding_dim)
        """
        h = self.input_proj(trace_sequence)       # (B, L, d_model)
        h = self.pos_encoder(h)                    # (B, L, d_model)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # Mean pooling (ignore padding)
        if src_key_padding_mask is not None:
            # Invert mask: True = valid token for averaging
            valid = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, L, 1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)                 # (B, d_model)

        return self.embed_proj(pooled)             # (B, embedding_dim)

    def forward(
        self,
        trace_sequence: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Full forward pass: embedding + classification logits.

        Parameters
        ----------
        trace_sequence : (batch, seq_len, n_features)

        Returns
        -------
        logits : (batch, n_classes)
        """
        embedding = self.encode(trace_sequence, src_key_padding_mask)
        return self.cls_head(embedding)

    def get_attention_weights(
        self,
        trace_sequence: Tensor,
    ) -> list:
        """
        Extract attention weight matrices from each encoder layer.
        Useful for visualisation in the Streamlit UI.

        Returns list of (batch, nhead, seq_len, seq_len) tensors.
        """
        h = self.input_proj(trace_sequence)
        h = self.pos_encoder(h)

        attn_weights = []
        for layer in self.encoder.layers:
            # Use the self-attention sub-layer directly
            attn_out, weights = layer.self_attn(
                h, h, h, need_weights=True, average_attn_weights=False
            )
            attn_weights.append(weights.detach())
            # Run the full layer to get correct h for next layer
            h = layer(h)

        return attn_weights
