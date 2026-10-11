"""
Analyze RTF embeddings to evaluate the 50/50 loss weighting.

Produces:
  1. UMAP visualization colored by class, group, anomaly score, LLM verdict
  2. Loss decomposition: correlation between recon_cont and recon_band
  3. Component-wise anomaly ranking vs LLM "interesting" verdicts
  4. Clustering metrics (silhouette, k-NN) on known-class sources

Usage:
    python analyze_embeddings.py \
        --embeddings fritz_embeddings.npz \
        --scores anomaly_scores.csv \
        --triage llm_triage_results.json \
        --manifest fritz_manifest.csv \
        --output-dir plots/
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(args):
    """Load and merge all data sources."""
    # Embeddings + decomposed losses
    emb_data = np.load(args.embeddings, allow_pickle=True)
    obj_ids = emb_data["obj_ids"]
    embeddings = emb_data["embeddings"]
    recon_cont = emb_data["recon_cont"]
    recon_band = emb_data["recon_band"]
    band_acc = emb_data["band_acc"]
    recon_error = emb_data["recon_error"]
    n_det = emb_data["n_detections"]

    df = pd.DataFrame(
        {
            "obj_id": obj_ids,
            "recon_cont": recon_cont,
            "recon_band": recon_band,
            "band_acc": band_acc,
            "recon_error": recon_error,
            "n_detections": n_det,
        }
    )

    # Anomaly scores
    scores_df = pd.read_csv(args.scores)
    df = df.merge(
        scores_df[["obj_id", "group_name", "classification", "anomaly_score", "if_score"]],
        on="obj_id",
        how="left",
    )

    # LLM triage results
    if args.triage and os.path.exists(args.triage):
        with open(args.triage) as f:
            triage = json.load(f)
        triage_df = pd.DataFrame([{"obj_id": r["obj_id"], "verdict": r["verdict"]} for r in triage])
        df = df.merge(triage_df, on="obj_id", how="left")
    else:
        df["verdict"] = None

    return df, embeddings


def plot_umap(embeddings, df, output_dir):
    """Generate UMAP visualizations colored by different attributes."""
    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed, trying sklearn TSNE instead...")
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        coords = reducer.fit_transform(embeddings)
        method = "t-SNE"
    else:
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        coords = reducer.fit_transform(embeddings)
        method = "UMAP"

    print(f"  {method} computed: {coords.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"RTF 128-dim Embedding Space ({method})", fontsize=16, fontweight="bold")

    # 1. Color by SkyPortal group
    ax = axes[0, 0]
    groups = df["group_name"].fillna("unknown")
    group_names = sorted(groups.unique())
    colors_map = {"Fast_Transients": "#e74c3c", "LIONS": "#2ecc71", "SLSNe": "#3498db"}
    for g in group_names:
        mask = groups == g
        c = colors_map.get(g, "#95a5a6")
        ax.scatter(coords[mask, 0], coords[mask, 1], c=c, s=8, alpha=0.6, label=g)
    ax.set_title("By SkyPortal Group", fontsize=12)
    ax.legend(fontsize=8, markerscale=2)

    # 2. Color by known classification
    ax = axes[0, 1]
    classes = df["classification"].fillna("unknown").replace("", "unknown")
    unique_classes = sorted(classes.unique())
    n_classes = len(unique_classes)
    cmap = plt.cm.get_cmap("tab20", n_classes)
    for i, cls in enumerate(unique_classes):
        mask = classes == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], s=8, alpha=0.6, label=cls)
    ax.set_title("By Known Classification", fontsize=12)
    if n_classes <= 15:
        ax.legend(fontsize=6, markerscale=2, ncol=2, loc="upper right")

    # 3. Color by anomaly score (heatmap)
    ax = axes[1, 0]
    scores = df["anomaly_score"].fillna(0).values
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="hot_r", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Anomaly Score")
    ax.set_title("By Anomaly Score", fontsize=12)

    # 4. Color by LLM verdict
    ax = axes[1, 1]
    verdicts = df["verdict"].fillna("not_reviewed")
    verdict_colors = {
        "interesting": "#e74c3c",
        "known_type": "#3498db",
        "artifact": "#95a5a6",
        "error": "#f39c12",
        "not_reviewed": "#dfe6e9",
    }
    for v in ["not_reviewed", "error", "artifact", "known_type", "interesting"]:
        mask = verdicts == v
        if mask.sum() > 0:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=verdict_colors.get(v, "#95a5a6"),
                s=12 if v == "interesting" else 6,
                alpha=0.8 if v in ("interesting", "known_type") else 0.3,
                label=f"{v} ({mask.sum()})",
                zorder=5 if v == "interesting" else 1,
            )
    ax.set_title("By LLM Verdict", fontsize=12)
    ax.legend(fontsize=8, markerscale=2)

    for ax in axes.flat:
        ax.set_xlabel(f"{method}-1", fontsize=9)
        ax.set_ylabel(f"{method}-2", fontsize=9)
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "umap_embeddings.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_loss_decomposition(df, output_dir):
    """Analyze correlation between recon_cont and recon_band."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Loss Component Analysis", fontsize=14, fontweight="bold")

    rc = df["recon_cont"].values
    rb = df["recon_band"].values

    # 1. Scatter: recon_cont vs recon_band
    ax = axes[0]
    corr = np.corrcoef(rc, rb)[0, 1]
    ax.scatter(rc, rb, s=6, alpha=0.4, c="#3498db")
    ax.set_xlabel("recon_cont (photometry MSE)", fontsize=10)
    ax.set_ylabel("recon_band (band CE)", fontsize=10)
    ax.set_title(f"Correlation: r = {corr:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. Component contribution to anomaly score
    ax = axes[1]
    total = rc + rb
    frac_cont = np.where(total > 0, rc / total, 0.5)
    ax.hist(frac_cont, bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="50/50 split")
    ax.set_xlabel("Fraction from recon_cont", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Loss Balance: cont / (cont + band)", fontsize=11)
    ax.legend(fontsize=9)

    # 3. Which component better predicts LLM "interesting"?
    ax = axes[2]
    reviewed = df[df["verdict"].isin(["interesting", "known_type"])].copy()
    if len(reviewed) > 5:
        reviewed["label"] = (reviewed["verdict"] == "interesting").astype(int)

        # Rank by each component
        for comp, color, label in [
            ("recon_cont", "#e74c3c", "recon_cont"),
            ("recon_band", "#3498db", "recon_band"),
            ("recon_error", "#2ecc71", "combined"),
        ]:
            ranked = reviewed.sort_values(comp, ascending=False)
            cumhits = ranked["label"].cumsum().values
            precision = cumhits / np.arange(1, len(cumhits) + 1)
            ax.plot(
                np.arange(1, len(precision) + 1),
                precision,
                color=color,
                label=f"{label}",
                lw=1.5,
            )

        ax.set_xlabel("Top-K (ranked by component)", fontsize=10)
        ax.set_ylabel("Precision @ K", fontsize=10)
        ax.set_title("Which loss catches 'interesting' better?", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Not enough\nLLM verdicts", ha="center", va="center", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "loss_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return corr


def compute_clustering_metrics(embeddings, df, output_dir):
    """Compute silhouette score and k-NN accuracy on known-class sources."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder

    # Filter to sources with known classification
    has_class = df["classification"].fillna("").replace("", np.nan).notna()
    if has_class.sum() < 10:
        print("  Too few classified sources for clustering metrics")
        return None, None

    idx = has_class.values
    X = embeddings[idx]
    labels = df.loc[idx, "classification"].values

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Silhouette score (needs at least 2 classes)
    n_classes = len(le.classes_)
    if n_classes >= 2:
        sil = silhouette_score(X, y, metric="cosine", sample_size=min(len(X), 5000))
    else:
        sil = float("nan")

    # k-NN accuracy (leave-one-out style with k=5)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(X) - 1), metric="cosine")
    from sklearn.model_selection import cross_val_score

    if len(X) >= 10:
        cv_scores = cross_val_score(knn, X, y, cv=min(5, n_classes), scoring="accuracy")
        knn_acc = cv_scores.mean()
    else:
        knn_acc = float("nan")

    print(f"  Classified sources: {has_class.sum()}")
    print(f"  Unique classes: {n_classes} ({', '.join(le.classes_)})")
    print(f"  Silhouette score: {sil:.4f}")
    print(f"  k-NN accuracy (5-fold CV): {knn_acc:.4f}")

    return sil, knn_acc


def print_summary(df, corr, sil, knn_acc, output_dir):
    """Print and save a summary report."""
    lines = [
        "# RTF Embedding Evaluation Summary",
        "",
        "## Loss Decomposition",
        f"- **recon_cont (photometry MSE):** mean={df['recon_cont'].mean():.6f}, std={df['recon_cont'].std():.6f}",
        f"- **recon_band (band CE):** mean={df['recon_band'].mean():.6f}, std={df['recon_band'].std():.6f}",
        f"- **Correlation (cont vs band):** r = {corr:.4f}",
        "",
    ]

    if corr > 0.8:
        lines.append(
            "> **High correlation.** Both components capture similar information. Weighting has minimal impact."
        )
    elif corr > 0.4:
        lines.append(
            "> **Moderate correlation.** Components are partially complementary. Current 50/50 is reasonable but could be tuned."
        )
    else:
        lines.append(
            "> **Low correlation.** Components capture different signals. One may be more useful for anomaly detection than the other."
        )

    lines += [
        "",
        "## Embedding Quality (known-class Fritz sources)",
        f"- **Silhouette score (cosine):** {sil:.4f}"
        if sil is not None
        else "- Silhouette: N/A (too few classes)",
        f"- **k-NN accuracy (5-fold CV, k=5):** {knn_acc:.4f}"
        if knn_acc is not None
        else "- k-NN: N/A",
        "",
        "## Interpretation",
    ]

    if sil is not None:
        if sil > 0.3:
            lines.append("- Silhouette > 0.3 indicates reasonable clustering by known class.")
        elif sil > 0.1:
            lines.append(
                "- Silhouette 0.1-0.3 indicates weak clustering. Embeddings partially separate types."
            )
        else:
            lines.append(
                "- Silhouette < 0.1 indicates poor clustering. Known types overlap heavily in embedding space."
            )

    lines += [
        "",
        "## Files",
        "- `umap_embeddings.png`: 4-panel UMAP visualization",
        "- `loss_decomposition.png`: loss component analysis + precision curves",
    ]

    report = "\n".join(lines)
    path = os.path.join(output_dir, "embedding_evaluation.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  Report: {path}")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze RTF embeddings")
    parser.add_argument("--embeddings", required=True, help="fritz_embeddings.npz")
    parser.add_argument("--scores", required=True, help="anomaly_scores.csv")
    parser.add_argument("--triage", default=None, help="llm_triage_results.json")
    parser.add_argument("--manifest", default=None, help="fritz_manifest.csv")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== RTF Embedding Evaluation ===")

    # Load data
    print("\n[1/4] Loading data...")
    df, embeddings = load_data(args)
    print(f"  {len(df)} sources, {embeddings.shape[1]}-dim embeddings")

    # UMAP visualization
    print("\n[2/4] Computing dimensionality reduction...")
    plot_umap(embeddings, df, args.output_dir)

    # Loss decomposition
    print("\n[3/4] Analyzing loss components...")
    corr = plot_loss_decomposition(df, args.output_dir)

    # Clustering metrics
    print("\n[4/4] Computing clustering metrics...")
    sil, knn_acc = compute_clustering_metrics(embeddings, df, args.output_dir)

    # Summary
    print_summary(df, corr, sil, knn_acc, args.output_dir)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
