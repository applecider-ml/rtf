"""
Compare two RTF embedding experiments side by side.

Produces a single figure with the key metrics from both models.

Usage:
    python compare_experiments.py \
        --baseline-embeddings ../fritz_embeddings.npz \
        --baseline-scores ../anomaly_scores.csv \
        --contonly-embeddings ../runs/ae_contonly/ae_dim128_meta_img/fritz_embeddings.npz \
        --contonly-scores ../runs/ae_contonly/ae_dim128_meta_img/anomaly_scores.csv \
        --triage ../llm_triage_results.json \
        --output-dir ../plots
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_experiment(emb_path, scores_path, triage_path=None):
    """Load embeddings + scores + optional triage verdicts."""
    emb = np.load(emb_path, allow_pickle=True)
    scores = pd.read_csv(scores_path)

    df = pd.DataFrame({
        "obj_id": emb["obj_ids"],
        "recon_cont": emb["recon_cont"],
        "recon_band": emb["recon_band"],
        "band_acc": emb["band_acc"],
        "recon_error": emb["recon_error"],
    })
    df = df.merge(
        scores[["obj_id", "anomaly_score", "if_score", "group_name", "classification"]],
        on="obj_id", how="left",
    )

    if triage_path and os.path.exists(triage_path):
        with open(triage_path) as f:
            triage = json.load(f)
        verdicts = {r["obj_id"]: r["verdict"] for r in triage}
        df["verdict"] = df["obj_id"].map(verdicts)
    else:
        df["verdict"] = None

    return df, emb["embeddings"]


def precision_at_k(df, col, max_k=None):
    """Compute precision@K for a given ranking column (higher = more anomalous)."""
    reviewed = df[df["verdict"].isin(["interesting", "known_type"])].copy()
    if len(reviewed) < 5:
        return np.array([]), np.array([])
    reviewed["label"] = (reviewed["verdict"] == "interesting").astype(int)
    ranked = reviewed.sort_values(col, ascending=False)
    cumhits = ranked["label"].cumsum().values
    k = max_k or len(cumhits)
    precision = cumhits[:k] / np.arange(1, k + 1)
    return np.arange(1, k + 1), precision


def main():
    parser = argparse.ArgumentParser(description="Compare two RTF experiments")
    parser.add_argument("--baseline-embeddings", required=True)
    parser.add_argument("--baseline-scores", required=True)
    parser.add_argument("--contonly-embeddings", required=True)
    parser.add_argument("--contonly-scores", required=True)
    parser.add_argument("--triage", default=None, help="LLM triage JSON (from baseline)")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Experiment Comparison ===")

    # Load both experiments
    print("\nLoading baseline (50/50)...")
    base_df, base_emb = load_experiment(
        args.baseline_embeddings, args.baseline_scores, args.triage
    )

    print("Loading cont-only (band_weight=0.0)...")
    cont_df, cont_emb = load_experiment(
        args.contonly_embeddings, args.contonly_scores, args.triage
    )

    # Clustering metrics
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    def cluster_metrics(emb, df, name):
        has_class = df["classification"].fillna("").replace("", np.nan).notna()
        if has_class.sum() < 10:
            return float("nan")
        X = emb[has_class.values]
        le = LabelEncoder()
        y = le.fit_transform(df.loc[has_class, "classification"].values)
        if len(set(y)) < 2:
            return float("nan")
        sil = silhouette_score(X, y, metric="cosine", sample_size=min(len(X), 5000))
        print(f"  {name}: silhouette={sil:.4f}, classes={len(set(y))}")
        return sil

    print("\nClustering metrics:")
    sil_base = cluster_metrics(base_emb, base_df, "Baseline (50/50)")
    sil_cont = cluster_metrics(cont_emb, cont_df, "Cont-only")

    # UMAP for both
    try:
        from umap import UMAP
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE as UMAP
        method = "t-SNE"

    print(f"\nComputing {method}...")
    def make_reducer():
        try:
            return UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        except TypeError:
            try:
                return UMAP(n_components=2, random_state=42)
            except TypeError:
                from sklearn.manifold import TSNE
                return TSNE(n_components=2, random_state=42, perplexity=30)

    reducer_base = make_reducer()
    coords_base = reducer_base.fit_transform(base_emb)
    reducer_cont = make_reducer()
    coords_cont = reducer_cont.fit_transform(cont_emb)

    # ── Main comparison figure ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("RTF Embedding Comparison: 50/50 vs Cont-Only", fontsize=16, fontweight="bold")

    # Row 0: Baseline
    # Row 1: Cont-only

    configs = [
        ("Baseline (50/50)", base_df, coords_base),
        ("Cont-Only (band_weight=0)", cont_df, coords_cont),
    ]

    for row, (label, df, coords) in enumerate(configs):
        # Column 0: by group
        ax = axes[row, 0]
        groups = df["group_name"].fillna("unknown")
        colors_map = {"Fast_Transients": "#e74c3c", "LIONS": "#2ecc71", "SLSNe": "#3498db"}
        for g in sorted(groups.unique()):
            mask = groups == g
            ax.scatter(coords[mask, 0], coords[mask, 1], c=colors_map.get(g, "#95a5a6"),
                      s=6, alpha=0.5, label=g)
        ax.set_title(f"{label}: By Group", fontsize=11)
        ax.legend(fontsize=7, markerscale=2)

        # Column 1: by anomaly score
        ax = axes[row, 1]
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                       c=df["anomaly_score"].fillna(0), cmap="hot_r", s=6, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="Score")
        ax.set_title(f"{label}: By Anomaly Score", fontsize=11)

        # Column 2: by LLM verdict
        ax = axes[row, 2]
        verdicts = df["verdict"].fillna("not_reviewed")
        v_colors = {"interesting": "#e74c3c", "known_type": "#3498db",
                    "not_reviewed": "#dfe6e9", "error": "#f39c12", "artifact": "#95a5a6"}
        for v in ["not_reviewed", "error", "artifact", "known_type", "interesting"]:
            mask = verdicts == v
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=v_colors.get(v, "#95a5a6"),
                          s=12 if v == "interesting" else 4,
                          alpha=0.8 if v == "interesting" else 0.2,
                          label=f"{v} ({mask.sum()})", zorder=5 if v == "interesting" else 1)
        ax.set_title(f"{label}: By LLM Verdict", fontsize=11)
        ax.legend(fontsize=7, markerscale=2)

    for ax in axes.flat:
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(args.output_dir, "comparison_umap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Precision@K comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, col, color, label in [
        (base_df, "recon_error", "#e74c3c", "Baseline: combined"),
        (base_df, "recon_cont", "#f39c12", "Baseline: recon_cont only"),
        (cont_df, "recon_error", "#2ecc71", "Cont-only: combined"),
        (cont_df, "recon_cont", "#3498db", "Cont-only: recon_cont"),
    ]:
        k, prec = precision_at_k(df, col)
        if len(k) > 0:
            ax.plot(k, prec, color=color, lw=2, label=label)

    ax.set_xlabel("Top-K", fontsize=11)
    ax.set_ylabel("Precision @ K", fontsize=11)
    ax.set_title("Which model catches 'interesting' better?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.output_dir, "comparison_precision.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Summary stats ──
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'Baseline':<15} {'Cont-Only':<15}")
    print("-" * 65)
    print(f"{'recon_cont mean':<35} {base_df['recon_cont'].mean():<15.4f} {cont_df['recon_cont'].mean():<15.4f}")
    print(f"{'recon_band mean':<35} {base_df['recon_band'].mean():<15.4f} {cont_df['recon_band'].mean():<15.4f}")
    print(f"{'band_acc mean':<35} {base_df['band_acc'].mean():<15.3f} {cont_df['band_acc'].mean():<15.3f}")
    print(f"{'Silhouette (cosine)':<35} {sil_base:<15.4f} {sil_cont:<15.4f}")
    print(f"{'anomaly_score range':<35} {base_df['anomaly_score'].min():.3f}-{base_df['anomaly_score'].max():.3f}       {cont_df['anomaly_score'].min():.3f}-{cont_df['anomaly_score'].max():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
