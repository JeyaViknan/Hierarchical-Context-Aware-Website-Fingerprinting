## Context-Aware Website Fingerprinting (WF) Demo

This repository contains an end-to-end **ML pipeline** for Website Fingerprinting on encrypted traffic metadata and a **Streamlit** app to demonstrate:

- **Baseline WF model**: classifies each traffic trace independently.
- **Context-aware WF model**: smooths predictions over a trace sequence (session) using a simple Viterbi decoder.
- **Intent inference**: maps the (context-smoothed) sequence to a human-readable intent using transparent rules.

### Dataset

This project expects a `.npy` file like `150sites.npy` in the workspace root.
The included dataset structure is parsed as:

- `dict[site] -> dict['quic'|'non-quic'] -> dict[trace_id] -> list[packet]`
- `packet := [proto, timestamp, signed_size]`

### Install

```bash
python3 -m pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

### Notes

- The dataset does not include user/session identifiers, so the app generates **synthetic sessions** by sampling traces at random. In a real system, sessions would be reconstructed from timestamps, user ids, or flow metadata.
- The "Website Category" is treated as the **site/domain** by default. You can extend the app to load a domain→category mapping and run intent rules on those categories.

