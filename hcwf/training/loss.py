"""
Loss functions for the HC-WF multi-task training pipeline.

Implements the combined loss:

    L = L_site + λ * L_intent

where:
  - L_site   = CrossEntropy with optional label smoothing (website classification)
  - L_intent = CrossEntropy with optional label smoothing (intent classification)
  - λ        = intent_loss_weight (configurable)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn, Tensor

from hcwf.utils.config import TrainingConfig


class MultitaskLoss(nn.Module):
    """
    Combined multi-task loss for website + intent classification.

    Parameters
    ----------
    intent_loss_weight : float (λ) — weight for the intent loss term
    label_smoothing    : float — label smoothing factor for both tasks
    n_sites            : int — number of website classes (for weighting)
    n_intents          : int — number of intent classes
    """

    def __init__(
        self,
        intent_loss_weight: float = 0.3,
        label_smoothing: float = 0.1,
        site_class_weights: Optional[Tensor] = None,
        intent_class_weights: Optional[Tensor] = None,
    ):
        super().__init__()
        self.intent_loss_weight = intent_loss_weight

        self.site_criterion = nn.CrossEntropyLoss(
            weight=site_class_weights,
            label_smoothing=label_smoothing,
        )

        self.intent_criterion = nn.CrossEntropyLoss(
            weight=intent_class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        site_logits: Tensor,
        site_labels: Tensor,
        intent_logits: Tensor,
        intent_labels: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute the combined multi-task loss.

        Parameters
        ----------
        site_logits   : (batch, n_sites)
        site_labels   : (batch,) integer labels
        intent_logits : (batch, n_intents)
        intent_labels : (batch,) integer labels

        Returns
        -------
        dict with keys:
          - "total"  : combined loss
          - "site"   : website classification loss
          - "intent" : intent classification loss
        """
        loss_site = self.site_criterion(site_logits, site_labels)
        loss_intent = self.intent_criterion(intent_logits, intent_labels)
        loss_total = loss_site + self.intent_loss_weight * loss_intent

        return {
            "total": loss_total,
            "site": loss_site.detach(),
            "intent": loss_intent.detach(),
        }


class Stage1Loss(nn.Module):
    """
    Simple cross-entropy loss for Stage 1 (packet transformer pre-training).
    """

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.criterion(logits, labels)
