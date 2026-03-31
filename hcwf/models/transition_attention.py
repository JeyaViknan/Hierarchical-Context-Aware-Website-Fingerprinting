"""
Transition-Aware Attention mechanism for the Session-Level Transformer.

This is the critical innovation of the HC-WF system. Standard
self-attention computes:

    attention_score = Q K^T / √d_k

Transition-Aware Attention augments this with a learnable bias matrix
T_ij that encodes the likelihood of browsing transitions between
positions in a session:

    attention_score = Q K^T / √d_k  +  T_ij

The T_ij matrix captures temporal browsing patterns:
  - Users tend to revisit recent pages (recency bias)
  - Certain website-type transitions are more common
  - Session position matters (early vs. late in session)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from hcwf.utils.config import TransitionAttentionConfig


class TransitionAwareAttention(nn.Module):
    """
    Multi-head attention with a learnable transition bias.

    For each head, the attention logit between positions i and j is:

        A_{ij} = (Q_i · K_j) / √d_k  +  T_{ij}

    where T_{ij} is a learnable per-head bias matrix.

    Parameters
    ----------
    d_model          : model dimension
    nhead            : number of attention heads
    max_session_len  : maximum number of traces in a session
    dropout          : attention dropout rate
    learnable_bias   : if True, T_ij is a learnable nn.Parameter;
                       if False, T_ij is a fixed distance-based decay
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_session_len: int = 5,
        dropout: float = 0.1,
        learnable_bias: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.max_session_len = max_session_len
        self.learnable_bias = learnable_bias
        self.scale = math.sqrt(self.d_k)

        # QKV projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Transition bias matrix T_ij
        if learnable_bias:
            # One bias matrix per head: (nhead, max_session_len, max_session_len)
            self.transition_bias = nn.Parameter(
                torch.zeros(nhead, max_session_len, max_session_len)
            )
            nn.init.normal_(self.transition_bias, mean=0.0, std=0.02)
        else:
            # Fixed distance-based decay bias
            positions = torch.arange(max_session_len).float()
            distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
            decay = torch.exp(-0.5 * distance)  # Exponential decay with distance
            self.register_buffer(
                "transition_bias",
                decay.unsqueeze(0).expand(nhead, -1, -1),
            )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        x    : (batch, seq_len, d_model) – sequence of trace embeddings
        mask : optional (batch, seq_len) boolean, True = valid position
        return_attention : if True, also return attention weights

        Returns
        -------
        output : (batch, seq_len, d_model)
        attn_weights : optional (batch, nhead, seq_len, seq_len)
        """
        B, L, D = x.shape
        assert L <= self.max_session_len, (
            f"Session length {L} exceeds max_session_len {self.max_session_len}"
        )

        # Project to Q, K, V
        Q = self.W_q(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.W_k(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # Standard scaled dot-product attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        # Add transition bias T_ij (the critical augmentation)
        T = self.transition_bias[:, :L, :L]  # (H, L, L) – crop to actual session length
        attn_logits = attn_logits + T.unsqueeze(0)  # broadcast over batch

        # Apply padding mask if provided
        if mask is not None:
            # mask shape: (B, L), True = valid
            # We need to mask out attention to padding positions
            padding_mask = ~mask  # True = padding
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_logits = attn_logits.masked_fill(padding_mask, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, d_k)

        # Concatenate heads and project
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, L, D)
        )
        output = self.resid_dropout(self.W_o(attn_output))

        if return_attention:
            return output, attn_weights.detach()
        return output, None

    def get_transition_bias(self) -> Tensor:
        """Return the current transition bias matrix for visualisation."""
        return self.transition_bias.detach().cpu()
