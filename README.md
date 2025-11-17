# Defense Conflict Analysis

This repository contains experiments for studying conflicts between safety and privacy defenses in large language models.  Each experiment script under `src/experiments` can be invoked as a standalone CLI.

## Conflict activation projection

The `conflict_activation_projection.py` CLI reproduces the cross-defense visualization described in the paper.  It forwards the malicious (safety) and privacy risk datasets through both the base model and the corresponding defense models, captures the hidden states at a target layer, and projects all activations into 2D using PCA, t-SNE, or UMAP before plotting the overlap.

### Usage

```bash
python -m src.experiments.conflict_activation_projection \
  --base <path/to/base/model> \
  --safety <path/to/safety/model> \
  --privacy <path/to/privacy/model> \
  --malicious <path/to/D_mal.jsonl> \
  --privacy-data <path/to/D_priv.jsonl> \
  --layer model.layers.16 \
  --projection pca \
  --batch-size 4 \
  --output results/conflict_projection
```

Key arguments:

- `--layer`: accepts a numeric layer index (e.g., `16`) or a full module path (e.g., `model.layers.16`).  The projection script only supports a single layer at a time.
- `--projection`: choose between `pca`, `tsne`, or `umap`.  PCA uses a shared `sklearn.decomposition.PCA(n_components=2)` fit for all activation groups.
- `--output`: directory where the script stores `conflict_projection.png` and the raw 2D coordinates as `projection_coords.npz` (containing base/defense splits for both safety and privacy datasets).

All dataset arguments follow the conventions from `03-extract_concepts.py`.  `build_dual_dataset` is used to load the JSONL corpora, so if you already have a `D_norm` path from earlier experiments you can supply it via `--normal` even though the projection script only consumes `D_mal` and `D_priv`.
