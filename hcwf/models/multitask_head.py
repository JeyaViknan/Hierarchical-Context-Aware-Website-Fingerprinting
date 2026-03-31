"""
Multi-task Classification Head for the HC-WF pipeline.

Takes the session representation h_session and produces predictions
for two tasks simultaneously:

  1. Website classification — identifies which website each trace
     in the session belongs to.
  2. Behavioral intent classification — identifies the user's
     browsing intent from the session pattern.

Architecture
------------
  h_session ──┬── SiteHead ──→ site logits (per-trace, from session context)
              │
              └── IntentHead ──→ intent logits (per-session)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn, Tensor

from hcwf.utils.config import MultitaskConfig


class SiteClassificationHead(nn.Module):
    """
    Website classification head.

    Maps the session representation to per-site logits.
    """

    def __init__(self, input_dim: int, n_sites: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_sites),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        logits : (batch, n_sites)
        """
        return self.head(x)


class IntentClassificationHead(nn.Module):
    """
    Behavioral intent classification head.

    Maps the session representation to intent logits.

    Intent categories (default):
      0 = Focused browsing (single site)
      1 = Comparison shopping (two sites)
      2 = Sequential exploration (multiple sites in order)
      3 = Looping / revisiting
      4 = Rapid switching
      5 = General browsing
    """

    INTENT_NAMES = [
        "Focused Browsing",
        "Comparison Shopping",
        "Sequential Exploration",
        "Looping / Revisiting",
        "Rapid Switching",
        "General Browsing",
    ]

    def __init__(self, input_dim: int, n_intents: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_intents),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        logits : (batch, n_intents)
        """
        return self.head(x)


class MultitaskHead(nn.Module):
    """
    Combined multi-task prediction head.

    Takes the session-level representation h_session and produces
    both website and intent predictions.

    Outputs
    -------
    dict with keys:
      - "site_logits"   : (batch, n_sites)
      - "intent_logits" : (batch, n_intents)
    """

    def __init__(self, cfg: MultitaskConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg

        self.site_head = SiteClassificationHead(
            input_dim=input_dim,
            n_sites=cfg.n_sites,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

        self.intent_head = IntentClassificationHead(
            input_dim=input_dim,
            n_intents=cfg.n_intents,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

    def forward(self, h_session: Tensor) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        h_session : (batch, input_dim) — session-level representation

        Returns
        -------
        dict with "site_logits" and "intent_logits"
        """
        return {
            "site_logits": self.site_head(h_session),
            "intent_logits": self.intent_head(h_session),
        }

    @staticmethod
    def get_intent_name(intent_id: int) -> str:
        """Map an intent integer to a human-readable name."""
        if 0 <= intent_id < len(IntentClassificationHead.INTENT_NAMES):
            return IntentClassificationHead.INTENT_NAMES[intent_id]
        return f"Unknown Intent ({intent_id})"
