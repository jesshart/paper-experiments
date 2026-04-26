# paper-experiments

Interactive [marimo](https://marimo.io) notebooks exploring concepts from
interpretability research papers.

## statistical_fragility.py

A visual companion to **["The Dead Salmons of AI
Interpretability"](https://arxiv.org/abs/2512.18792)** (Méloux, Dirupo, Portet
& Peyrard, 2025). Drag sliders and watch noise dress itself up as a finding
across feature attribution, probing, causal discovery, sparse autoencoders,
concept-based explanations, and mechanistic circuit search.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_bZxdKEPewrPVg3cin6oi7V)

### Run locally

The notebook declares its own dependencies via inline PEP 723 metadata,
so [`uv`](https://docs.astral.sh/uv/) can run it in an isolated sandbox
without touching your project env:

```bash
uvx marimo edit --sandbox statistical_fragility.py
```
