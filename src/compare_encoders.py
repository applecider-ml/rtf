"""
Compare Encoders: RTF vs Perceiver IO vs AstroDistill.

Loads all three trained models, extracts 128D embeddings from the same
sharded dataset, and produces:
  1. Side-by-side UMAP visualization (1×3 panel)
  2. Silhouette scores (cosine) per encoder
  3. k-NN classification accuracy (k=10, 5-fold CV)
  4. Linear probe accuracy (logistic regression, 5-fold CV)
  5. Summary comparison table (JSON + printed)

All three models use the same embed() interface:
    embedding = model.embed(x, pad_mask, images)  # (B, 128)

Usage:
    python compare_encoders.py \
        --data-dir /tmp/shards \
        --rtf-model   runs/rtf/best_model.pt \
        --perc-model  runs/perceiver/best_model.pt \
        --astro-model runs/astrodistill/best_model.pt \
        --output-dir  runs/comparison
"""

import argparse
import json
import os
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────


def load_rtf(model_path, device):
    """Load the RTF baseline (LightCurveCompressor)."""
    from model import LightCurveCompressor

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_kwargs = checkpoint.get("model_kwargs", {})
    else:
        state_dict = checkpoint
        model_kwargs = {}

    if not model_kwargs:
        model_kwargs = {
            "mode": "ae",
            "latent_dim": 128,
            "in_channels": 37,
            "d_model": 256,
            "use_images": True,
        }

    model = LightCurveCompressor(**model_kwargs)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()
    return model


def load_perceiver(model_path, device):
    """Load the Perceiver IO masked autoencoder."""
    from perceiver_model import PerceiverAutoencoder

    model = PerceiverAutoencoder(
        phot_channels=7,
        meta_channels=30,
        d_model=256,
        n_latents=32,
        n_cross=2,
        n_self=6,
        n_heads=8,
        dropout=0.1,
        embed_dim=128,
        mask_ratio=0.50,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_astrodistill(model_path, device):
    """Load the AstroDistill self-distillation encoder."""
    from astrodistill_model import AstroDistill

    model = AstroDistill(
        phot_channels=7,
        meta_channels=30,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        embed_dim=128,
        mask_ratio=0.50,
        top_k_layers=2,
        pred_hidden=512,
        ema_decay=0.999,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# ──────────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def extract_embeddings(model, fritz_dir, scores_path, device):
    """Extract 128D embeddings from Fritz NPZ files and merge labels.

    Args:
        model: Any model with .embed(x, pad_mask, images) -> (B, 128)
        fritz_dir: Path to Fritz sources NPZ dir
        scores_path: Path to anomaly_scores.csv containing labels
        device: torch device

    Returns:
        dict with 'embeddings' (N, 128), 'labels_fine' (N,), 'obj_ids' (N,)
    """
    all_emb, all_ids = [], []
    npz_files = sorted(Path(fritz_dir).glob("*.npz"))
    npz_files = [f for f in npz_files if f.stem != "fritz_manifest"]

    MAX_LEN = 257

    for npz_path in npz_files:
        obj_id = npz_path.stem
        try:
            data = np.load(npz_path, allow_pickle=True)
            x_np = data["x"]
            image_np = data["images"]

            if x_np.shape[0] < 3:
                continue

            if x_np.shape[0] > MAX_LEN:
                x_np = x_np[:MAX_LEN]

            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            pad_mask = torch.zeros(1, x.shape[1], dtype=torch.bool).to(device)
            images = torch.from_numpy(image_np).unsqueeze(0).to(device)

            emb = model.embed(x, pad_mask, images)  # (1, 128)
            all_emb.append(emb[0].cpu().numpy())
            all_ids.append(obj_id)
        except Exception:
            continue

    # Merge labels
    df = pd.DataFrame({"obj_id": all_ids})
    scores_df = pd.read_csv(scores_path)
    df = df.merge(
        scores_df[["obj_id", "classification"]],
        on="obj_id",
        how="left",
    )

    # Filter only known classifications
    has_class = df["classification"].fillna("").replace("", np.nan).notna()
    df_known = df[has_class].copy()

    # CRITICAL FIX: The raw classification column has dozens of rare classes with
    # only 1 or 2 examples. This mathematically breaks Silhouette score and
    # cross-validation. We will filter to only the top 5 most common classes.
    top_5_classes = df_known["classification"].value_counts().nlargest(5).index
    is_top_5 = df_known["classification"].isin(top_5_classes)
    df_filtered = df_known[is_top_5]

    # Get the indices of the filtered sources relative to the original all_ids list
    idx = df_filtered.index.values
    embeddings_arr = np.array(all_emb)[idx]
    labels_arr = df_filtered["classification"].values
    obj_ids_arr = df_filtered["obj_id"].values

    return {
        "embeddings": embeddings_arr,
        "labels_fine": labels_arr,
        "obj_ids": obj_ids_arr,
    }


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────


def compute_metrics(embeddings, labels, encoder_name):
    """Compute clustering and classification metrics on embeddings.

    Returns dict with silhouette, knn_acc, linear_probe_acc, n_classes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Filter out unlabeled samples (label == -1 or 0 depending on convention)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    X = embeddings
    results = {"encoder": encoder_name, "n_samples": len(X), "n_classes": n_classes}

    # Silhouette score (cosine distance)
    if n_classes >= 2:
        # Subsample for speed if necessary
        n_sil = min(len(X), 10000)
        idx = np.random.choice(len(X), n_sil, replace=False)
        results["silhouette"] = float(silhouette_score(X[idx], y[idx], metric="cosine"))
    else:
        results["silhouette"] = float("nan")

    # k-NN accuracy (k=10, 5-fold CV, cosine distance)
    if n_classes >= 2 and len(X) >= 20:
        knn = KNeighborsClassifier(
            n_neighbors=min(10, len(X) // 5),
            metric="cosine",
        )
        knn_scores = cross_val_score(
            knn,
            X,
            y,
            cv=min(5, n_classes),
            scoring="accuracy",
        )
        results["knn_acc"] = float(knn_scores.mean())
        results["knn_std"] = float(knn_scores.std())
    else:
        results["knn_acc"] = float("nan")
        results["knn_std"] = float("nan")

    # Linear probe (logistic regression, 5-fold CV)
    if n_classes >= 2 and len(X) >= 20:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            C=1.0,
            n_jobs=-1,
        )
        lp_scores = cross_val_score(
            clf,
            X_scaled,
            y,
            cv=min(5, n_classes),
            scoring="accuracy",
        )
        results["linear_probe_acc"] = float(lp_scores.mean())
        results["linear_probe_std"] = float(lp_scores.std())
    else:
        results["linear_probe_acc"] = float("nan")
        results["linear_probe_std"] = float("nan")

    return results


# ──────────────────────────────────────────────────────────────────────
# UMAP visualization
# ──────────────────────────────────────────────────────────────────────


def build_class_colormap(all_labels):
    """Create a consistent colormap for class labels across all panels."""
    unique = sorted(set(all_labels))
    n = len(unique)
    cmap = plt.cm.get_cmap("tab20", max(n, 2))
    return {cls: cmap(i) for i, cls in enumerate(unique)}


def plot_comparison_umap(encoder_data, output_dir):
    """Generate 1×3 UMAP comparison plot.

    Args:
        encoder_data: list of (name, embeddings, labels_fine) tuples
        output_dir: save path
    """
    try:
        from umap import UMAP

        method = "UMAP"
        reducer_cls = UMAP
    except ImportError:
        from sklearn.manifold import TSNE

        method = "t-SNE"
        reducer_cls = None

    n_encoders = len(encoder_data)

    # Collect all labels for consistent colormap
    all_labels = []
    for _, _, labels in encoder_data:
        all_labels.extend(labels.tolist())
    colormap = build_class_colormap(all_labels)
    unique_labels = sorted(colormap.keys())

    fig, axes = plt.subplots(
        1,
        n_encoders,
        figsize=(7 * n_encoders, 6),
        constrained_layout=True,
    )
    if n_encoders == 1:
        axes = [axes]

    fig.suptitle(
        f"128D Embedding Space Comparison ({method})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    for ax, (name, emb, labels) in zip(axes, encoder_data):
        print(f"  Computing {method} for {name}...")
        t0 = time.time()

        # Subsample for speed
        n_plot = min(len(emb), 20000)
        idx = np.random.choice(len(emb), n_plot, replace=False)
        emb_sub = emb[idx]
        labels_sub = labels[idx]

        if reducer_cls is not None:
            reducer = UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.3,
                random_state=42,
            )
        else:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)

        coords = reducer.fit_transform(emb_sub)
        dt = time.time() - t0
        print(f"    {method} done in {dt:.1f}s")

        # Plot each class
        for cls in unique_labels:
            mask = labels_sub == cls
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[colormap[cls]],
                s=6,
                alpha=0.6,
            )

        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel(f"{method}-1", fontsize=10)
        ax.set_ylabel(f"{method}-2", fontsize=10)
        ax.tick_params(labelsize=8)

    # Shared legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colormap[cls],
            markersize=8,
            label=f"{cls}",
        )
        for cls in unique_labels
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(len(unique_labels), 8),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )

    path = os.path.join(output_dir, "encoder_comparison_umap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare RTF vs Perceiver IO vs AstroDistill embeddings"
    )
    parser.add_argument("--fritz-dir", required=True, help="Path to Fritz sources NPZ dir")
    parser.add_argument("--scores", required=True, help="Path to anomaly_scores.csv")
    parser.add_argument("--rtf-model", default=None, help="RTF best_model.pt path")
    parser.add_argument("--perc-model", default=None, help="Perceiver best_model.pt path")
    parser.add_argument("--astro-model", default=None, help="AstroDistill best_model.pt path")
    parser.add_argument("--output-dir", default="runs/comparison", help="Output dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  ENCODER COMPARISON: RTF vs Perceiver IO vs AstroDistill")
    print("=" * 70)

    # --- Build model registry ---
    models = []  # list of (name, load_fn, path)
    if args.rtf_model and os.path.exists(args.rtf_model):
        models.append(("RTF (Baseline)", load_rtf, args.rtf_model))
    if args.perc_model and os.path.exists(args.perc_model):
        models.append(("Perceiver IO", load_perceiver, args.perc_model))
    if args.astro_model and os.path.exists(args.astro_model):
        models.append(("AstroDistill", load_astrodistill, args.astro_model))

    if not models:
        print("ERROR: No model paths provided or found. Nothing to compare.")
        return

    print(f"\n  Models to compare: {len(models)}")
    for name, _, path in models:
        print(f"    - {name}: {path}")

    # --- Extract embeddings for each model ---
    encoder_results = []  # list of (name, emb_data, metrics)
    encoder_umap_data = []  # for UMAP plot

    for i, (name, load_fn, path) in enumerate(models):
        step = i + 2
        print(f"\n[{step}/{len(models) + 2}] {name}")

        # Load
        print(f"  Loading model from {path}...")
        t0 = time.time()
        model = load_fn(path, args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded: {n_params:,} parameters ({time.time() - t0:.1f}s)")

        # Extract embeddings
        print(f"  Extracting embeddings from {args.fritz_dir}...")
        t0 = time.time()
        emb_data = extract_embeddings(model, args.fritz_dir, args.scores, args.device)
        dt = time.time() - t0
        emb = emb_data["embeddings"]
        print(f"  Extracted: {emb.shape} in {dt:.1f}s")

        # Save individual embeddings
        emb_path = os.path.join(
            args.output_dir,
            f"embeddings_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.npz",
        )
        np.savez_compressed(emb_path, **emb_data)
        print(f"  Saved: {emb_path}")

        # Compute metrics
        print("  Computing metrics...")
        metrics = compute_metrics(
            emb_data["embeddings"],
            emb_data["labels_fine"],
            name,
        )
        metrics["n_params"] = n_params
        encoder_results.append(metrics)

        print(f"    Silhouette:    {metrics['silhouette']:.4f}")
        print(f"    k-NN (k=10):   {metrics['knn_acc']:.4f} ± {metrics['knn_std']:.4f}")
        print(
            f"    Linear Probe:  {metrics['linear_probe_acc']:.4f} ± {metrics['linear_probe_std']:.4f}"
        )

        # Collect for UMAP
        encoder_umap_data.append((name, emb_data["embeddings"], emb_data["labels_fine"]))

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # --- UMAP visualization ---
    final_step = len(models) + 2
    print(f"\n[{final_step}/{final_step}] Generating UMAP comparison plot...")
    plot_comparison_umap(encoder_umap_data, args.output_dir)

    # --- Print comparison table ---
    print(f"\n{'=' * 90}")
    print("ENCODER COMPARISON RESULTS")
    print(f"{'=' * 90}")
    print(f"{'Encoder':<20} {'Params':>10} {'Silhouette':>12} {'k-NN Acc':>10} {'LinProbe':>10}")
    print("-" * 70)
    for r in encoder_results:
        print(
            f"{r['encoder']:<20} {r['n_params']:>10,} "
            f"{r['silhouette']:>12.4f} "
            f"{r['knn_acc']:>10.4f} "
            f"{r['linear_probe_acc']:>10.4f}"
        )
    print(f"{'=' * 90}")

    # --- Save results ---
    results_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(encoder_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # --- Determine winner ---
    best = max(encoder_results, key=lambda r: r.get("silhouette", -1))
    print(f"\n  🏆 Best Silhouette: {best['encoder']} ({best['silhouette']:.4f})")
    best_knn = max(encoder_results, key=lambda r: r.get("knn_acc", -1))
    print(f"  🏆 Best k-NN:       {best_knn['encoder']} ({best_knn['knn_acc']:.4f})")
    best_lp = max(encoder_results, key=lambda r: r.get("linear_probe_acc", -1))
    print(f"  🏆 Best LinProbe:   {best_lp['encoder']} ({best_lp['linear_probe_acc']:.4f})")

    print(f"\n{'=' * 70}")
    print("  COMPARISON COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
