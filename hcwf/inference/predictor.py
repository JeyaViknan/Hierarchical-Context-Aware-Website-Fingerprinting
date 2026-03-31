"""
Inference predictor for the HC-WF pipeline.

Provides a high-level API for end-to-end prediction:

  raw traces → preprocess → Stage 1 embeddings → build sessions →
  Stage 2 context → Multi-task predictions

The Predictor class encapsulates the full two-stage pipeline and
can be used for both single-session and batch inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from hcwf.data.preprocessing import preprocess_trace, preprocess_batch
from hcwf.data.session_builder import build_sessions_synthetic, collate_sessions
from hcwf.models.packet_transformer import PacketTransformer
from hcwf.models.session_transformer import SessionTransformer
from hcwf.models.multitask_head import MultitaskHead, IntentClassificationHead
from hcwf.utils.config import HCWFConfig

logger = logging.getLogger(__name__)


class HCWFPredictor:
    """
    End-to-end inference for the Hierarchical Context-Aware WF system.

    Usage
    -----
    >>> predictor = HCWFPredictor(cfg)
    >>> predictor.load_models(stage1_path, stage2_path)
    >>> result = predictor.predict_session(raw_traces)
    """

    def __init__(
        self,
        cfg: HCWFConfig,
        site_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.site_names = site_names or [f"Site_{i}" for i in range(cfg.multitask.n_sites)]
        self.device = torch.device(device or cfg.training.resolve_device())

        # Models (initialised but untrained)
        self.packet_transformer = PacketTransformer(
            n_features=cfg.preprocess.n_features,
            n_classes=cfg.multitask.n_sites,
            cfg=cfg.packet_transformer,
        ).to(self.device)

        self.session_transformer = SessionTransformer(
            cfg=cfg.session_transformer,
            transition_cfg=cfg.transition_attention,
        ).to(self.device)

        self.multitask_head = MultitaskHead(
            cfg=cfg.multitask,
            input_dim=cfg.session_transformer.d_model,
        ).to(self.device)

    def load_models(
        self,
        stage1_path: Optional[str] = None,
        stage2_path: Optional[str] = None,
    ) -> None:
        """Load trained checkpoints for both stages."""
        if stage1_path:
            ckpt = torch.load(stage1_path, map_location=self.device, weights_only=True)
            self.packet_transformer.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded Stage 1 from {stage1_path}")

        if stage2_path:
            ckpt = torch.load(stage2_path, map_location=self.device, weights_only=True)
            self.session_transformer.load_state_dict(ckpt["session_model_state_dict"])
            self.multitask_head.load_state_dict(ckpt["multitask_head_state_dict"])
            logger.info(f"Loaded Stage 2 from {stage2_path}")

    @torch.no_grad()
    def predict_traces(
        self,
        raw_traces: List[np.ndarray],
        timestamps: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1 only: classify individual traces.

        Parameters
        ----------
        raw_traces : list of raw packet-size arrays
        timestamps : optional list of timestamp arrays

        Returns
        -------
        dict with:
          - "site_predictions" : list of predicted site names
          - "site_probabilities" : (N, n_sites) array
          - "embeddings" : (N, embedding_dim) tensor
        """
        self.packet_transformer.eval()

        X = preprocess_batch(raw_traces, self.cfg.preprocess, timestamps)
        X = X.to(self.device)

        logits = self.packet_transformer(X)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        embeddings = self.packet_transformer.encode(X).cpu()

        return {
            "site_predictions": [self.site_names[p] for p in preds],
            "site_probabilities": probs,
            "site_indices": preds,
            "embeddings": embeddings,
        }

    @torch.no_grad()
    def predict_session(
        self,
        raw_traces: List[np.ndarray],
        timestamps: Optional[List[np.ndarray]] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Full two-stage inference on a session of traces.

        Flow:
          1. Preprocess traces
          2. Generate embeddings (Stage 1)
          3. Build session
          4. Run session transformer (Stage 2)
          5. Output website + intent predictions

        Parameters
        ----------
        raw_traces       : list of raw packet-size arrays (one session)
        timestamps       : optional list of timestamp arrays
        return_attention : whether to return attention weights

        Returns
        -------
        dict with:
          - "site_prediction"      : predicted website name
          - "site_probabilities"   : (n_sites,) array
          - "intent_prediction"    : predicted intent name
          - "intent_probabilities" : (n_intents,) array
          - "per_trace_sites"      : list of per-trace site predictions
          - "attention_weights"    : optional list of attention matrices
          - "transition_biases"    : optional list of T_ij matrices
        """
        self.packet_transformer.eval()
        self.session_transformer.eval()
        self.multitask_head.eval()

        # Step 1: Preprocess
        X = preprocess_batch(raw_traces, self.cfg.preprocess, timestamps)
        X = X.to(self.device)

        # Step 2: Generate trace embeddings
        embeddings = self.packet_transformer.encode(X)  # (L, embedding_dim)
        trace_logits = self.packet_transformer(X)
        per_trace_preds = trace_logits.argmax(dim=-1).cpu().numpy()

        # Step 3: Build session tensor
        L = embeddings.shape[0]
        max_L = self.cfg.session.max_session_len
        session_emb = torch.zeros(1, max_L, embeddings.shape[1], device=self.device)
        mask = torch.zeros(1, max_L, dtype=torch.bool, device=self.device)

        actual_len = min(L, max_L)
        session_emb[0, :actual_len] = embeddings[:actual_len]
        mask[0, :actual_len] = True

        # Step 4: Session transformer
        h_session, attn_weights = self.session_transformer(
            session_emb, mask=mask, return_attention=return_attention
        )

        # Step 5: Multi-task prediction
        predictions = self.multitask_head(h_session)

        site_probs = torch.softmax(predictions["site_logits"], dim=-1).cpu().numpy()[0]
        intent_probs = torch.softmax(predictions["intent_logits"], dim=-1).cpu().numpy()[0]
        site_pred = int(np.argmax(site_probs))
        intent_pred = int(np.argmax(intent_probs))

        result: Dict[str, Any] = {
            "site_prediction": self.site_names[site_pred],
            "site_index": site_pred,
            "site_probabilities": site_probs,
            "intent_prediction": MultitaskHead.get_intent_name(intent_pred),
            "intent_index": intent_pred,
            "intent_probabilities": intent_probs,
            "per_trace_sites": [self.site_names[p] for p in per_trace_preds],
            "per_trace_indices": per_trace_preds,
            "session_embedding": h_session.cpu().numpy()[0],
        }

        if return_attention:
            result["attention_weights"] = [
                w.cpu().numpy() if w is not None else None for w in attn_weights
            ]
            result["transition_biases"] = [
                b.numpy() for b in self.session_transformer.get_all_transition_biases()
            ]

        return result

    @torch.no_grad()
    def predict_batch_sessions(
        self,
        sessions: List[List[np.ndarray]],
        timestamps_list: Optional[List[List[np.ndarray]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference on multiple sessions."""
        results = []
        for i, session_traces in enumerate(sessions):
            ts = timestamps_list[i] if timestamps_list else None
            results.append(self.predict_session(session_traces, ts))
        return results

    def get_model_summary(self) -> Dict[str, int]:
        """Return parameter counts for all model components."""
        def count_params(model):
            return sum(p.numel() for p in model.parameters())

        return {
            "packet_transformer_params": count_params(self.packet_transformer),
            "session_transformer_params": count_params(self.session_transformer),
            "multitask_head_params": count_params(self.multitask_head),
            "total_params": (
                count_params(self.packet_transformer)
                + count_params(self.session_transformer)
                + count_params(self.multitask_head)
            ),
        }
