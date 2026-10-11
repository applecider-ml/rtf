# RTF Embedding Evaluation Summary

## Loss Decomposition
- **recon_cont (photometry MSE):** mean=3.385720, std=1.966088
- **recon_band (band CE):** mean=4.052473, std=1.887328
- **Correlation (cont vs band):** r = 0.1001

> **Low correlation.** Components capture different signals. One may be more useful for anomaly detection than the other.

## Embedding Quality (known-class Fritz sources)
- **Silhouette score (cosine):** -0.8672
- **k-NN accuracy (5-fold CV, k=5):** 0.2465

## Interpretation
- Silhouette < 0.1 indicates poor clustering. Known types overlap heavily in embedding space.

## Files
- `umap_embeddings.png`: 4-panel UMAP visualization
- `loss_decomposition.png`: loss component analysis + precision curves