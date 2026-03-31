"""
HC-WF Streamlit Frontend
=========================

Interactive web UI for the Hierarchical Context-Aware Website
Fingerprinting system.  Provides:

  • Session upload / mock generation
  • Live website + intent predictions
  • Attention weight visualisation
  • Session flow diagram
  • Transition bias heatmaps
  • Model architecture overview
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from hcwf.utils.config import HCWFConfig
from hcwf.data.preprocessing import preprocess_trace, generate_dummy_traces
from hcwf.data.session_builder import generate_dummy_sessions
from hcwf.models.packet_transformer import PacketTransformer
from hcwf.models.session_transformer import SessionTransformer
from hcwf.models.multitask_head import MultitaskHead, IntentClassificationHead
from hcwf.inference.predictor import HCWFPredictor


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HC-WF | Hierarchical Context-Aware Website Fingerprinting",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium design ────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Gradient header */
    .hcwf-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hcwf-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hcwf-header p {
        color: #a5b4fc;
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.25);
    }
    .metric-card .label {
        color: #a5b4fc;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-card .value {
        color: #e0e7ff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* Prediction pill */
    .prediction-pill {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
        margin: 0.25rem 0.15rem;
    }

    .intent-pill {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        margin: 0.25rem 0.15rem;
    }

    /* Stage badge */
    .stage-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .stage-1 {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .stage-2 {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Section divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
        margin: 2rem 0;
    }

    /* Architecture diagram styling */
    .arch-block {
        background: rgba(30, 27, 75, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        font-size: 0.85rem;
        color: #c7d2fe;
    }

    /* Dark scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #4c1d95; border-radius: 3px; }

    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────────

@st.cache_resource
def init_predictor(config_dict: dict) -> HCWFPredictor:
    """Initialise the HC-WF predictor (cached across reruns)."""
    cfg = HCWFConfig.from_dict(config_dict)
    return HCWFPredictor(cfg)


def generate_mock_traces(n_traces: int, seed: int) -> List[np.ndarray]:
    """Generate mock raw traces for demo purposes."""
    rng = np.random.RandomState(seed)
    traces = []
    for i in range(n_traces):
        n_pkt = rng.randint(150, 2000)
        # Create somewhat realistic patterns
        out_ratio = 0.3 + 0.4 * rng.random()
        directions = rng.choice([1, -1], size=n_pkt, p=[out_ratio, 1 - out_ratio])
        magnitudes = rng.exponential(600 + 100 * i, size=n_pkt)
        traces.append((directions * magnitudes).astype(np.float32))
    return traces


def make_attention_heatmap(
    weights: np.ndarray,
    title: str,
    head_idx: int = 0,
) -> go.Figure:
    """Create a Plotly heatmap from attention weights."""
    if weights.ndim == 4:
        w = weights[0, head_idx]  # first batch, selected head
    elif weights.ndim == 3:
        w = weights[head_idx]
    else:
        w = weights

    fig = go.Figure(data=go.Heatmap(
        z=w,
        colorscale=[
            [0.0, "#0f0c29"],
            [0.25, "#302b63"],
            [0.5, "#4f46e5"],
            [0.75, "#7c3aed"],
            [1.0, "#c4b5fd"],
        ],
        showscale=True,
        colorbar=dict(title="Weight", tickfont=dict(color="#a5b4fc")),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e7ff", size=14)),
        xaxis=dict(
            title="Key Position",
            color="#a5b4fc",
            gridcolor="rgba(139) 92, 246, 0.1)",
        ),
        yaxis=dict(
            title="Query Position",
            color="#a5b4fc",
            autorange="reversed",
        ),
        plot_bgcolor="rgba(15, 12, 41, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a5b4fc"),
        height=350,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


def make_session_flow(
    per_trace_sites: List[str],
    intent: str,
) -> go.Figure:
    """Create a session flow visualisation."""
    n = len(per_trace_sites)
    x_positions = list(range(n))

    # Assign colors based on unique sites
    unique_sites = list(dict.fromkeys(per_trace_sites))
    palette = px.colors.qualitative.Vivid
    site_colors = {site: palette[i % len(palette)] for i, site in enumerate(unique_sites)}

    fig = go.Figure()

    # Connection lines
    for i in range(n - 1):
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i + 1]],
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(139, 92, 246, 0.4)", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Trace nodes
    for i, site in enumerate(per_trace_sites):
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[0],
            mode="markers+text",
            marker=dict(
                size=40,
                color=site_colors[site],
                line=dict(color="white", width=2),
                symbol="circle",
            ),
            text=[f"T{i+1}"],
            textposition="middle center",
            textfont=dict(color="white", size=11, family="Inter"),
            name=site,
            showlegend=True,
            hovertemplate=f"<b>Trace {i+1}</b><br>Website: {site}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"Session Flow → Intent: {intent}",
            font=dict(color="#e0e7ff", size=14),
        ),
        xaxis=dict(
            title="Trace Order",
            color="#a5b4fc",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            visible=False,
            range=[-1, 1],
        ),
        plot_bgcolor="rgba(15, 12, 41, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a5b4fc"),
        height=250,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
    )
    return fig


def make_probability_bar(
    probs: np.ndarray,
    names: List[str],
    title: str,
    top_k: int = 10,
) -> go.Figure:
    """Create a horizontal bar chart of top-k prediction probabilities."""
    indices = np.argsort(probs)[::-1][:top_k]
    top_names = [names[i] if i < len(names) else f"Class {i}" for i in indices]
    top_probs = probs[indices]

    fig = go.Figure(go.Bar(
        x=top_probs[::-1],
        y=top_names[::-1],
        orientation="h",
        marker=dict(
            color=top_probs[::-1],
            colorscale=[
                [0.0, "#312e81"],
                [0.5, "#4f46e5"],
                [1.0, "#8b5cf6"],
            ],
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e7ff", size=14)),
        xaxis=dict(title="Probability", color="#a5b4fc", range=[0, 1]),
        yaxis=dict(color="#a5b4fc"),
        plot_bgcolor="rgba(15, 12, 41, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a5b4fc"),
        height=max(250, top_k * 30),
        margin=dict(l=120, r=20, t=50, b=40),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("**Model Architecture**")
    d_model = st.select_slider(
        "Model dimension",
        options=[64, 96, 128, 192, 256],
        value=128,
        help="Embedding dimension for both stages",
    )
    n_heads = st.select_slider("Attention heads", options=[2, 4, 8], value=4)
    n_layers_s1 = st.select_slider("Stage 1 layers", options=[1, 2, 3, 4], value=3)
    n_layers_s2 = st.select_slider("Stage 2 layers", options=[1, 2, 3], value=2)

    st.markdown("---")
    st.markdown("**Data Settings**")
    n_sites = st.slider("Number of sites", 10, 200, 100, 10)
    n_intents = st.slider("Number of intents", 3, 10, 6)
    max_session_len = st.slider("Max session length", 2, 8, 5)

    st.markdown("---")
    st.markdown("**Session Demo**")
    session_size = st.slider("Demo session traces", 2, 5, 3)
    demo_seed = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")
    st.markdown("**Transition Attention**")
    learnable_bias = st.checkbox("Learnable T_ij", value=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#6b7280; font-size:0.75rem;'>"
        "HC-WF v0.1.0 · Research Prototype</div>",
        unsafe_allow_html=True,
    )


# ── Build config from sidebar ───────────────────────────────────────────────

config_dict = {
    "preprocess": {"max_trace_len": 1000, "n_features": 2},
    "session": {"max_session_len": max_session_len},
    "packet_transformer": {
        "d_model": d_model, "nhead": n_heads,
        "num_layers": n_layers_s1, "dim_feedforward": d_model * 2,
        "embedding_dim": d_model,
    },
    "session_transformer": {
        "d_model": d_model, "nhead": n_heads,
        "num_layers": n_layers_s2, "dim_feedforward": d_model * 2,
        "max_session_len": max_session_len,
    },
    "transition_attention": {
        "max_session_len": max_session_len,
        "learnable_bias": learnable_bias,
    },
    "multitask": {
        "n_sites": n_sites, "n_intents": n_intents,
        "hidden_dim": d_model,
    },
    "training": {"device": "cpu"},
}


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hcwf-header">
    <h1>🔐 HC-WF: Hierarchical Context-Aware Website Fingerprinting</h1>
    <p>Two-stage Transformer pipeline with Transition-Aware Attention · Multi-task Learning</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_predict, tab_arch, tab_attention, tab_config = st.tabs([
    "🔮 Prediction", "🏗️ Architecture", "👁️ Attention Visualisation", "📋 Configuration",
])


# ── Tab 1: Prediction ───────────────────────────────────────────────────────

with tab_predict:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<span class="stage-badge stage-1">Stage 1</span> → '
                    '<span class="stage-badge stage-2">Stage 2</span>',
                    unsafe_allow_html=True)
        st.markdown("#### Input Session")

        input_method = st.radio(
            "Trace input method",
            ["🎲 Generate mock traces", "📁 Upload trace file"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if input_method == "📁 Upload trace file":
            uploaded = st.file_uploader(
                "Upload a CSV / NPY trace file",
                type=["csv", "npy"],
                help="Each row is a trace: comma-separated signed packet sizes",
            )
            if uploaded is not None:
                if uploaded.name.endswith(".npy"):
                    data = np.load(uploaded, allow_pickle=True)
                    raw_traces = [data[i].astype(np.float32) for i in range(min(len(data), session_size))]
                else:
                    import io, csv
                    content = uploaded.read().decode("utf-8")
                    reader = csv.reader(io.StringIO(content))
                    raw_traces = []
                    for row in reader:
                        if row and len(raw_traces) < session_size:
                            raw_traces.append(np.array([float(x) for x in row], dtype=np.float32))
                st.success(f"Loaded {len(raw_traces)} traces from file")
            else:
                raw_traces = generate_mock_traces(session_size, int(demo_seed))
                st.info("No file uploaded — using mock traces")
        else:
            raw_traces = generate_mock_traces(session_size, int(demo_seed))

        # Show trace summaries
        for i, trace in enumerate(raw_traces):
            n_out = int(np.sum(trace > 0))
            n_in = int(np.sum(trace < 0))
            st.markdown(
                f"**Trace {i+1}** — {len(trace)} packets "
                f"(↑{n_out} outgoing, ↓{n_in} incoming)"
            )

    with col_right:
        st.markdown("#### Predictions")

        # Initialise predictor
        predictor = init_predictor(config_dict)

        with st.spinner("Running two-stage inference..."):
            t0 = time.time()
            result = predictor.predict_session(raw_traces, return_attention=True)
            inference_time = time.time() - t0

        # Display predictions
        st.markdown(
            f'<div class="prediction-pill">🌐 {result["site_prediction"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="intent-pill">🧠 {result["intent_prediction"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Metric cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Inference</div>
                <div class="value">{inference_time*1000:.0f}ms</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Traces</div>
                <div class="value">{len(raw_traces)}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            top_prob = float(np.max(result["site_probabilities"]))
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Confidence</div>
                <div class="value">{top_prob:.1%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Session flow
    st.markdown("#### 📊 Session Flow")
    flow_fig = make_session_flow(result["per_trace_sites"], result["intent_prediction"])
    st.plotly_chart(flow_fig, use_container_width=True)

    # Probability distributions
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        site_names = [f"Site_{i}" for i in range(n_sites)]
        fig = make_probability_bar(
            result["site_probabilities"],
            site_names,
            "Top-10 Website Predictions",
            top_k=10,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_prob2:
        intent_names = IntentClassificationHead.INTENT_NAMES[:n_intents]
        fig = make_probability_bar(
            result["intent_probabilities"],
            intent_names,
            "Intent Predictions",
            top_k=n_intents,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Architecture ─────────────────────────────────────────────────────

with tab_arch:
    st.markdown("#### 🏗️ HC-WF Architecture Overview")

    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                    HC-WF: Two-Stage Transformer Pipeline                │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ┌─────────────────────── STAGE 1 ───────────────────────┐              │
    │  │  Packet-Level Transformer (Trace Encoder)             │              │
    │  │                                                        │              │
    │  │  Raw Trace ─→ [Linear Proj] ─→ [Pos. Encoding]       │              │
    │  │      ─→ [N × TransformerEncoder] ─→ [Mean Pool]      │              │
    │  │      ─→ z_j (trace embedding)                         │              │
    │  └────────────────────────────────────────────────────────┘              │
    │                           │                                              │
    │                    z_1, z_2, ..., z_L                                    │
    │                           │                                              │
    │  ┌─────────────────────── STAGE 2 ───────────────────────┐              │
    │  │  Session-Level Transformer (Context Encoder)          │              │
    │  │                                                        │              │
    │  │  [z_1..z_L] ─→ [Session Pos. Enc.]                   │              │
    │  │      ─→ [N × TransitionAwareEncoderBlock]             │              │
    │  │            └─ Attention = QK^T/√d + T_ij   ← NOVEL   │              │
    │  │      ─→ [Mean Pool] ─→ h_session                      │              │
    │  └────────────────────────────────────────────────────────┘              │
    │                           │                                              │
    │                    h_session                                             │
    │                     ┌─────┴─────┐                                        │
    │               ┌─────┴─────┐ ┌───┴────┐                                  │
    │               │ Site Head │ │ Intent │                                   │
    │               │ (softmax) │ │  Head  │                                   │
    │               └───────────┘ └────────┘                                   │
    │                     │            │                                        │
    │               Website ID    Intent Class                                 │
    └──────────────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Model summary
    st.markdown("#### 📊 Model Summary")
    summary = predictor.get_model_summary()
    sum_cols = st.columns(4)
    labels = ["Packet Transformer", "Session Transformer", "Multi-task Head", "Total"]
    keys = list(summary.keys())
    for col, label, key in zip(sum_cols, labels, keys):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{summary[key]:,}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### 🔑 Key Innovation: Transition-Aware Attention")
    st.markdown("""
    Standard self-attention: `A = softmax(QK^T / √d_k)`

    **Transition-Aware Attention** adds a learnable bias:

    `A = softmax(QK^T / √d_k + T_ij)`

    Where **T_ij** is a learnable matrix that captures:
    - **Recency bias**: Users tend to revisit recently browsed sites
    - **Transition patterns**: Certain site-type pairs co-occur frequently
    - **Position effects**: Early vs. late browsing within a session
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📐 Training Pipeline")
    st.markdown("""
    | Phase | Description | Details |
    |-------|-------------|---------|
    | **Stage 1** | Train Packet Transformer | Independent per-trace website classification |
    | **Freeze** | Lock Stage 1 weights | Encoder becomes a fixed feature extractor |
    | **Stage 2** | Train Session Transformer | Context modeling + multi-task heads |
    | **Loss** | `L = L_site + λ · L_intent` | Cross-entropy with label smoothing |
    """)


# ── Tab 3: Attention Visualisation ───────────────────────────────────────────

with tab_attention:
    st.markdown("#### 👁️ Attention Weight Visualisation")

    if result.get("attention_weights"):
        n_available_layers = len(result["attention_weights"])

        attn_col1, attn_col2 = st.columns([1, 3])
        with attn_col1:
            layer_idx = st.selectbox(
                "Layer",
                range(n_available_layers),
                format_func=lambda x: f"Layer {x + 1}",
            )
            head_idx = st.selectbox(
                "Head",
                range(n_heads),
                format_func=lambda x: f"Head {x + 1}",
            )

        with attn_col2:
            weights = result["attention_weights"][layer_idx]
            if weights is not None:
                fig = make_attention_heatmap(
                    weights,
                    f"Session Attention — Layer {layer_idx + 1}, Head {head_idx + 1}",
                    head_idx=head_idx,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Attention weights not available for this layer.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Transition bias visualisation
        st.markdown("#### 🔄 Learned Transition Bias (T_ij)")
        if result.get("transition_biases"):
            bias_cols = st.columns(n_available_layers)
            for l_idx, (col, bias) in enumerate(zip(bias_cols, result["transition_biases"])):
                with col:
                    fig = make_attention_heatmap(
                        bias,
                        f"T_ij — Layer {l_idx + 1}",
                        head_idx=min(head_idx, bias.shape[0] - 1),
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Run a prediction from the Prediction tab first to generate attention weights."
        )


# ── Tab 4: Configuration ────────────────────────────────────────────────────

with tab_config:
    st.markdown("#### 📋 Current Configuration")
    cfg_display = HCWFConfig.from_dict(config_dict)
    st.code(cfg_display.to_yaml(), language="yaml")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📂 System Information")
    info_data = {
        "Device": config_dict["training"]["device"],
        "PyTorch Version": torch.__version__,
        "Python Version": sys.version.split()[0],
        "CUDA Available": str(torch.cuda.is_available()),
    }
    for k, v in info_data.items():
        st.markdown(f"**{k}**: `{v}`")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📦 Dataset Integration")
    st.info(
        "**Dataset will be integrated later.** The current system supports "
        "plug-and-play dataset loading. To integrate your dataset:\n\n"
        "1. Implement a loader in `hcwf/data/preprocessing.py`\n"
        "2. Map your format to `List[np.ndarray]` of signed packet sizes\n"
        "3. Provide labels as integer indices\n"
        "4. Update config `n_sites` and `n_intents` accordingly"
    )
