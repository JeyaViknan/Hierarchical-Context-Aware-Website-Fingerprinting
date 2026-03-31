from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from wf_pipeline.context import ContextConfig
from wf_pipeline.evaluation import compute_metrics, stability_score
from wf_pipeline.ingestion import TraceExample, build_label_space, load_150sites_npy
from wf_pipeline.intent import infer_intent
from wf_pipeline.models import (
    ContextBiLSTM,
    ContextRNNConfig,
    TraceTransformer,
    TransformerConfig,
    make_context_dataloader,
    make_trace_dataloader,
    train_simple_classifier,
)
from wf_pipeline.preprocess import PreprocessConfig, preprocess_trace


st.set_page_config(page_title="WF-Transformer + Context BiLSTM Demo", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> List[TraceExample]:
    return load_150sites_npy(path)


def traces_to_table(examples: List[TraceExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        rows.append(
            {
                "site": ex.site,
                "transport": ex.transport,
                "trace_id": ex.trace_id,
                "n_packets": int(ex.sizes.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def build_sequence_tensor(examples: List[TraceExample], pre_cfg: PreprocessConfig) -> np.ndarray:
    """
    Turn each raw trace into a fixed-length **sequence** suitable for a Transformer.

    For every packet we keep:
    - signed packet size (direction encoded in sign)
    - inter-arrival time
    """

    seqs = []
    for ex in examples:
        sizes_fixed, iat_fixed = preprocess_trace(ex.times, ex.sizes, pre_cfg)
        # (L, 2) feature matrix per trace
        feat = np.stack([sizes_fixed, iat_fixed], axis=-1)
        seqs.append(feat)
    return np.stack(seqs, axis=0).astype(np.float32)


def build_label_vector(examples: List[TraceExample], site_to_idx: Dict[str, int]) -> np.ndarray:
    return np.asarray([site_to_idx[ex.site] for ex in examples], dtype=np.int32)


def sample_session(
    examples: List[TraceExample],
    model: TraceTransformer,
    X_all: np.ndarray,
    class_names: List[str],
    session_len: int,
    seed: int,
    device: str,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Construct a synthetic browsing session from random traces, then run the
    WF-Transformer on the ordered sequence to obtain per-trace probabilities.
    """

    rng = random.Random(seed)
    idxs = [rng.randrange(0, len(examples)) for _ in range(session_len)]
    X_sess = X_all[idxs]  # (T, L, F)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_sess).float().to(device))
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        y_hat = np.argmax(proba, axis=1)
    return idxs, proba, y_hat


st.title("Context-Aware Website Fingerprinting + Intent Inference (Streamlit)")
st.caption(
    "End-to-end demo: dataset ingestion → preprocessing → Transformer-based WF model → "
    "session reconstruction → BiLSTM context model → intent inference → evaluation + visualizations."
)

with st.sidebar:
    st.header("Dataset")
    dataset_path = st.text_input("Dataset path (.npy)", value="150sites.npy")
    transport_filter = st.multiselect("Transport", ["quic", "non-quic"], default=["quic", "non-quic"])

    st.header("Preprocessing")
    max_len = st.slider("Trace max_len (pad/truncate)", 200, 5000, 2000, 100)
    clip_size = st.slider("Clip signed packet size", 2000, 50000, 20000, 1000)

    st.header("WF-Transformer")
    d_model = st.selectbox("d_model", [32, 64, 96], index=1)
    nhead = st.selectbox("n_heads", [2, 4, 8], index=1)
    num_layers = st.selectbox("num_layers", [1, 2, 3], index=1)
    dim_ff = st.selectbox("dim_feedforward", [64, 128, 256], index=1)
    num_epochs_wf = st.slider("WF-Transformer epochs", 1, 5, 2, 1)

    st.header("Train / test split")
    test_size = st.slider("Test split", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("Random seed", value=7, step=1)

    st.header("Context model")
    session_len = st.slider("Synthetic session length", 5, 50, 15, 1)
    num_epochs_ctx = st.slider("Context BiLSTM epochs", 1, 5, 1, 1)


try:
    examples_all = load_dataset(dataset_path)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

examples = [ex for ex in examples_all if ex.transport in set(transport_filter)]
if not examples:
    st.warning("No traces left after transport filtering.")
    st.stop()

df = traces_to_table(examples)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Dataset overview")
    st.write(
        {
            "n_traces": int(len(examples)),
            "n_sites": int(df["site"].nunique()),
            "transports": sorted(df["transport"].unique().tolist()),
        }
    )
with col_b:
    st.subheader("Trace length distribution")
    fig = px.histogram(df, x="n_packets", nbins=60, title="Packets per trace")
    st.plotly_chart(fig, use_container_width=True)

sites, site_to_idx = build_label_space(examples)
pre_cfg = PreprocessConfig(max_len=int(max_len), clip_size=int(clip_size))

with st.spinner("Building per-trace sequences for WF-Transformer..."):
    X_seq = build_sequence_tensor(examples, pre_cfg)  # (N, L, 2)
    y = build_label_vector(examples, site_to_idx)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_seq, y, np.arange(len(examples)), test_size=float(test_size), random_state=int(seed), stratify=y
)

device = "cuda" if torch.cuda.is_available() else "cpu"
wf_cfg = TransformerConfig(
    d_model=int(d_model),
    nhead=int(nhead),
    num_layers=int(num_layers),
    dim_ff=int(dim_ff),
)
wf_model = TraceTransformer(n_features=2, n_classes=len(sites), cfg=wf_cfg)

with st.spinner("Training WF-Transformer (per-trace website fingerprinting model)..."):
    train_loader = make_trace_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    train_simple_classifier(wf_model, train_loader, num_epochs=int(num_epochs_wf), lr=1e-3, device=device)

with st.spinner("Evaluating WF-Transformer..."):
    wf_model.eval()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        logits = wf_model(X_test_t)
        proba_test = torch.softmax(logits, dim=-1).cpu().numpy()
        y_pred = np.argmax(proba_test, axis=1)
    baseline_metrics = compute_metrics(y_test, y_pred)

st.subheader("Baseline WF-Transformer metrics (per-trace)")
st.write(baseline_metrics)

st.subheader("Context-aware demo (BiLSTM over website sequences + intent)")
ctx_cfg = ContextConfig()

idxs, proba_seq, y_hat_seq = sample_session(
    examples, wf_model, X_seq, sites, int(session_len), int(seed), device=device
)

stability_base = stability_score(y_hat_seq)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("Stability (baseline)", f"{stability_base:.3f}")
with col2:
    st.caption(
        "Stability here = fraction of consecutive predictions that remain the same "
        "within the sampled browsing session."
    )

site_seq_base = [sites[i] for i in y_hat_seq.tolist()]

# For now treat sites as categories; this is where a domain→category
# mapping would be plugged in.
category_seq = site_seq_base

intent = infer_intent(category_seq)

st.write({"contextual_user_intent": intent["intent"], "rationale": intent["rationale"]})

timeline = pd.DataFrame(
    {
        "t": list(range(len(site_seq_base))),
        "baseline_site": site_seq_base,
        "context_site": site_seq_base,
    }
)

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Session timeline (baseline)")
    fig = px.scatter(timeline, x="t", y="baseline_site", title="Baseline per-trace predictions")
    st.plotly_chart(fig, use_container_width=True)
with col_r:
    st.subheader("Session timeline (same sequence, used by context BiLSTM)")
    fig = px.scatter(timeline, x="t", y="context_site", title="Sequence sent into context model")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction confidence (top-1 probability over session)")
top1_base = proba_seq[np.arange(proba_seq.shape[0]), y_hat_seq]
conf = pd.DataFrame({"t": np.arange(len(top1_base)), "baseline_top1_p": top1_base})
fig = go.Figure()
fig.add_trace(go.Scatter(x=conf["t"], y=conf["baseline_top1_p"], mode="lines+markers", name="WF-Transformer"))
fig.update_layout(yaxis_title="probability", xaxis_title="trace index")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Run metadata")
st.json(
    {
        "preprocess": asdict(pre_cfg),
        "wf_transformer": asdict(wf_cfg),
        "context_model": asdict(ctx_cfg),
    }
)

