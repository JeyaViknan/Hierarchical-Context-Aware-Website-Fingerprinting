# 🔐 HC-WF: Hierarchical Context-Aware Website Fingerprinting

A **production-quality research prototype** implementing a two-stage Transformer pipeline for website fingerprinting on encrypted traffic, with session-level context modeling and multi-task learning.

---

## 📖 Project Overview

**HC-WF** upgrades traditional per-trace website fingerprinting into a hierarchical system that models both packet-level patterns and session-level browsing context.

### Key Innovations

1. **Hierarchical Two-Stage Architecture** — Stage 1 encodes individual traffic traces; Stage 2 models inter-trace context within browsing sessions.
2. **Transition-Aware Attention** — Augments standard self-attention with a learnable bias matrix **T_ij** that captures browsing transition patterns:
   ```
   attention_score = QK^T / √d_k  +  T_ij
   ```
3. **Multi-Task Learning** — Jointly predicts both the visited website and the user's behavioral intent from session patterns.
4. **Session Modeling** — Groups individual traces into browsing sessions using time-gap logic, enabling context-aware prediction.

---

## 🏗️ Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    HC-WF: Two-Stage Transformer Pipeline                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────── STAGE 1 ───────────────────────┐              │
│  │  Packet-Level Transformer (Trace Encoder)             │              │
│  │                                                        │              │
│  │  Raw Trace ─→ [Linear Proj + LayerNorm + GELU]        │              │
│  │      ─→ [Relative Positional Encoding]                │              │
│  │      ─→ [N × Pre-Norm TransformerEncoder]             │              │
│  │      ─→ [Mean Pool] ─→ [Projection + LayerNorm]      │              │
│  │      ─→ z_j (trace embedding)                         │              │
│  └────────────────────────────────────────────────────────┘              │
│                           │                                              │
│               z_1, z_2, ..., z_L  (session embeddings)                  │
│                           │                                              │
│  ┌─────────────────────── STAGE 2 ───────────────────────┐              │
│  │  Session-Level Transformer (Context Encoder)          │              │
│  │                                                        │              │
│  │  [z_1..z_L] ─→ [Input Proj + LayerNorm]              │              │
│  │      ─→ [Session Positional Encoding]                 │              │
│  │      ─→ [N × TransitionAwareEncoderBlock]             │              │
│  │            ├─ Pre-Norm + TransitionAwareAttention      │              │
│  │            │    └─ A = QK^T/√d + T_ij  ← NOVEL       │              │
│  │            └─ Pre-Norm + FFN                           │              │
│  │      ─→ [Final LayerNorm] ─→ [Mean Pool]             │              │
│  │      ─→ h_session                                     │              │
│  └────────────────────────────────────────────────────────┘              │
│                           │                                              │
│                    h_session                                             │
│                     ┌─────┴─────┐                                        │
│               ┌─────┴─────┐ ┌───┴──────┐                                │
│               │ Site Head │ │ Intent   │                                 │
│               │  (MLP +   │ │  Head    │                                 │
│               │  softmax) │ │ (MLP +   │                                 │
│               │           │ │ softmax) │                                 │
│               └───────────┘ └──────────┘                                 │
│                     │            │                                        │
│               Website ID    Intent Class                                 │
│                                                                          │
│  Loss: L = L_site + λ · L_intent  (CrossEntropy + label smoothing)      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Hierarchical-Context-Aware-Website-Fingerprinting.git
cd Hierarchical-Context-Aware-Website-Fingerprinting

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run the full pipeline (uses dummy data)
python -m hcwf.main --demo

# Run with custom configuration
python -m hcwf.main --config config.yaml

# Launch the Streamlit UI
streamlit run hcwf/app/streamlit_app.py
```

---

## 🏋️ Training Instructions

### Stage 1: Packet-Level Transformer

Train the trace encoder independently on per-trace website classification:

```bash
# Train Stage 1 only
python -m hcwf.main --stage 1

# With custom settings
python -m hcwf.main --stage 1 --device cuda --seed 123
```

**What happens:**
1. Raw packet traces are converted to fixed-length feature tensors (direction + IAT)
2. The Packet Transformer learns to classify individual traces
3. Encoder weights are saved to `checkpoints/stage1/`

### Stage 2: Session-Level Transformer

Train the context encoder with frozen Stage 1 weights:

```bash
# Train Stage 2 only (requires Stage 1 checkpoint)
python -m hcwf.main --stage 2
```

**What happens:**
1. Stage 1 encoder is frozen (no gradient updates)
2. Trace embeddings are generated using the frozen encoder
3. The Session Transformer with Transition-Aware Attention is trained
4. Multi-task heads (site + intent) are optimised jointly
5. Loss: `L = L_site + λ × L_intent`

### Full Pipeline

```bash
# Train both stages sequentially
python -m hcwf.main

# Quick demo with minimal epochs
python -m hcwf.main --demo
```

### Configuration

Create a YAML config file to customise all hyperparameters:

```yaml
preprocess:
  max_trace_len: 2000
  clip_size: 20000
  include_timing: true

packet_transformer:
  d_model: 128
  nhead: 4
  num_layers: 3
  dim_feedforward: 256
  embedding_dim: 128

session_transformer:
  d_model: 128
  nhead: 4
  num_layers: 2
  max_session_len: 5

transition_attention:
  max_session_len: 5
  learnable_bias: true

multitask:
  n_sites: 100
  n_intents: 6

training:
  stage1_epochs: 30
  stage2_epochs: 20
  stage1_lr: 0.001
  stage2_lr: 0.0005
  intent_loss_weight: 0.3
  label_smoothing: 0.1
```

---

## 📁 Project Structure

```
Hierarchical-Context-Aware-Website-Fingerprinting/
│
├── hcwf/                              # Main HC-WF package
│   │
│   ├── data/                          # Data pipeline
│   │   ├── preprocessing.py           # Trace → tensor conversion, dummy data gen
│   │   └── session_builder.py         # Group traces into sessions
│   │
│   ├── models/                        # Neural network architectures
│   │   ├── packet_transformer.py      # Stage 1: Packet-Level Transformer
│   │   ├── session_transformer.py     # Stage 2: Session-Level Transformer
│   │   ├── transition_attention.py    # Transition-Aware Attention (T_ij)
│   │   └── multitask_head.py          # Site + Intent classification heads
│   │
│   ├── training/                      # Training pipelines
│   │   ├── train_stage1.py            # Stage 1 training loop
│   │   ├── train_stage2.py            # Stage 2 training loop (frozen Stage 1)
│   │   └── loss.py                    # Multi-task loss functions
│   │
│   ├── inference/                     # Inference pipeline
│   │   └── predictor.py              # End-to-end HCWFPredictor class
│   │
│   ├── utils/                         # Utilities
│   │   ├── config.py                  # Centralized configuration (YAML/JSON)
│   │   └── metrics.py                 # Evaluation metrics (Acc, F1, AUC, etc.)
│   │
│   ├── app/                           # Frontend
│   │   └── streamlit_app.py           # Interactive Streamlit UI
│   │
│   └── main.py                        # CLI entry point
│
├── wf_pipeline/                       # Legacy pipeline (original codebase)
│
├── checkpoints/                       # Saved model weights (auto-created)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `preprocessing.py` | Converts raw packet sizes → `(max_len, 2)` tensors with direction and IAT features |
| `session_builder.py` | Groups traces into sessions using time-gap logic or synthetic sampling |
| `packet_transformer.py` | Multi-head attention over packet sequences → trace embedding `z_j` |
| `session_transformer.py` | Context encoder over `(z_1, ..., z_L)` → session representation `h_session` |
| `transition_attention.py` | **Core innovation**: `A = QK^T/√d + T_ij` with learnable transition bias |
| `multitask_head.py` | Dual-head MLP for website classification + intent classification |
| `loss.py` | Combined loss `L = L_site + λ · L_intent` with label smoothing |
| `train_stage1.py` | Stage 1 training with AdamW, cosine LR, early stopping, checkpointing |
| `train_stage2.py` | Stage 2 training with frozen encoder and multi-task optimisation |
| `predictor.py` | High-level API: raw traces → (website, intent) predictions |
| `config.py` | Hierarchical dataclass config with YAML/JSON serialization |
| `metrics.py` | Accuracy, Precision, Recall, F1, ROC-AUC, session accuracy |

---

## 🖥️ Streamlit UI

The interactive frontend provides:

- **🔮 Prediction Tab** — Upload traces or generate mock data; see website + intent predictions with confidence metrics and session flow visualization
- **🏗️ Architecture Tab** — System overview, parameter counts, and the transition-aware attention explanation
- **👁️ Attention Tab** — Interactive heatmaps of attention weights and learned transition bias matrices T_ij
- **📋 Config Tab** — Live YAML configuration display and system information

```bash
streamlit run hcwf/app/streamlit_app.py
```

---

## 📊 Metrics

The system evaluates using:

| Metric | Scope | Description |
|--------|-------|-------------|
| Accuracy | Per-trace / Per-session | Fraction of correct predictions |
| Precision (macro) | Per-class | Average per-class precision |
| Recall (macro) | Per-class | Average per-class recall |
| F1 (macro / weighted) | Per-class | Harmonic mean of precision and recall |
| ROC-AUC (OvR) | Per-class | One-vs-Rest area under the ROC curve |
| Stability | Session | Fraction of consistent consecutive predictions |
| Session Accuracy | Session | All traces in a session must be correct |

---

## 🔮 Future Dataset Integration

> **Dataset will be integrated later. The current system supports plug-and-play dataset loading.**

To integrate your own dataset:

1. **Implement a data loader** that returns `List[np.ndarray]` of signed packet sizes (positive=outgoing, negative=incoming)
2. **Provide labels** as integer arrays (site IDs and intent IDs)
3. **Update config** with your class counts (`n_sites`, `n_intents`)
4. **Optionally provide timestamps** for real session construction (otherwise synthetic sessions are generated)

The system is designed to work with any packet-level traffic trace dataset, including:
- AWF (Automated Website Fingerprinting)
- CW (Closed-World) datasets
- OW (Open-World) datasets
- Custom captures from tools like `tshark` or `tcpdump`

---

## 🧠 Technical Details

### Transition-Aware Attention

The key innovation is augmenting standard dot-product attention with a position-dependent learnable bias:

```
Standard:     A_ij = (Q_i · K_j) / √d_k
HC-WF:        A_ij = (Q_i · K_j) / √d_k + T_ij
```

Each attention head has its own `T_ij ∈ ℝ^{L×L}` matrix that learns:
- **Temporal decay**: Nearby traces attend more strongly  
- **Transition affinities**: Certain position pairs have higher attention
- **Session structure**: Start-of-session and end-of-session patterns

### Multi-Task Learning

The combined loss function encourages the model to learn representations that are informative for both tasks:

```
L = L_site + λ · L_intent
```

where `λ` (default 0.3) controls the relative importance of intent classification.

---

## 📄 License

This project is a research prototype. See LICENSE for details.

---

## 🙏 Acknowledgments

Built upon concepts from website fingerprinting research, Transformer architectures, and multi-task learning in network traffic analysis.
