"""
Compare Encoders v2: Full 10-Model Grid Evaluation.

Loads all trained model variants, extracts embeddings from Fritz labeled
sources, and produces:
  1. Silhouette scores (cosine) per encoder
  2. k-NN classification accuracy (k=10, 5-fold CV)
  3. Linear probe accuracy (logistic regression, 5-fold CV)
  4. Per-class F1 scores for the top model vs RTF baseline
  5. Confusion matrices for top model vs RTF
  6. UMAP visualization (top-4 models)
  7. Summary comparison table (JSON + printed)

Usage:
    python compare_encoders_v2.py \
        --fritz-dir /path/to/fritz_sources \
        --scores /path/to/anomaly_scores.csv \
        --runs-dir /path/to/rtf/runs \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ======================================================================
# MODEL REGISTRY — every model we want to compare
# ======================================================================

MODEL_REGISTRY = [
    # --- Phase 1 baselines ---
    {
        "name": "RTF 128-D",
        "arch": "rtf",
        "embed_dim": 128,
        "backbone": "simple",
        "run_dir": "ae_dim128_meta_img",
        "group": "baseline",
    },
    {
        "name": "Perceiver 128-D",
        "arch": "perceiver",
        "embed_dim": 128,
        "backbone": "simple",
        "run_dir": "perceiver",
        "group": "baseline",
    },
    {
        "name": "AstroDistill 128-D",
        "arch": "astrodistill",
        "embed_dim": 128,
        "backbone": "simple",
        "run_dir": "astrodistill",
        "group": "baseline",
    },
    # --- Dimensionality sweep (flatlined but best checkpoints saved) ---
    {
        "name": "Perceiver 32-D",
        "arch": "perceiver",
        "embed_dim": 32,
        "backbone": "simple",
        "run_dir": "perceiver_32d",
        "group": "dim_sweep",
    },
    {
        "name": "Perceiver 64-D",
        "arch": "perceiver",
        "embed_dim": 64,
        "backbone": "simple",
        "run_dir": "perceiver_64d",
        "group": "dim_sweep",
    },
    {
        "name": "AstroDistill 32-D",
        "arch": "astrodistill",
        "embed_dim": 32,
        "backbone": "simple",
        "run_dir": "astrodistill_32d",
        "group": "dim_sweep",
    },
    {
        "name": "AstroDistill 64-D",
        "arch": "astrodistill",
        "embed_dim": 64,
        "backbone": "simple",
        "run_dir": "astrodistill_64d",
        "group": "dim_sweep",
    },
    # --- Backbone experiments ---
    {
        "name": "AstroDistill+Zoobot 128-D",
        "arch": "astrodistill",
        "embed_dim": 128,
        "backbone": "zoobot",
        "run_dir": "astrodistill_zoobot",
        "group": "backbone",
    },
    {
        "name": "AstroDistill+DINOv2 128-D",
        "arch": "astrodistill",
        "embed_dim": 128,
        "backbone": "dinov2",
        "run_dir": "astrodistill_dinov2",
        "group": "backbone",
    },
    # --- The combo run ---
    {
        "name": "AstroDistill+Zoobot 64-D",
        "arch": "astrodistill",
        "embed_dim": 64,
        "backbone": "zoobot",
        "run_dir": "astrodistill_zoobot_64d",
        "group": "combo",
    },
    # --- Phase 2: Zoobot Large backbone + latent dim experiments ---
    {
        "name": "RTF+ZooL 64-D",
        "arch": "rtf",
        "embed_dim": 64,
        "backbone": "zoobot_large",
        "run_dir": "rtf_zoobot_large_64d/ae_dim64_meta_img",
        "group": "phase2",
    },
    {
        "name": "RTF+ZooL 128-D",
        "arch": "rtf",
        "embed_dim": 128,
        "backbone": "zoobot_large",
        "run_dir": "rtf_zoobot_large_128d/ae_dim128_meta_img",
        "group": "phase2",
    },
    {
        "name": "RTF 64-D",
        "arch": "rtf",
        "embed_dim": 64,
        "backbone": "simple",
        "d_model": 128,
        "in_channels": 7,
        "use_images": False,
        "run_dir": "ae_dim64",
        "group": "phase2",
    },
    # --- Phase 3: Zoobot Pico + DINOv2 backbone experiments ---
    {
        "name": "RTF+ZooP 64-D",
        "arch": "rtf",
        "embed_dim": 64,
        "backbone": "zoobot",
        "run_dir": "rtf_zoobot_pico_64d/ae_dim64_meta_img",
        "group": "phase3",
    },
    {
        "name": "RTF+ZooP 128-D",
        "arch": "rtf",
        "embed_dim": 128,
        "backbone": "zoobot",
        "run_dir": "rtf_zoobot_pico_128d/ae_dim128_meta_img",
        "group": "phase3",
    },
    {
        "name": "RTF+DINOv2 128-D",
        "arch": "rtf",
        "embed_dim": 128,
        "backbone": "dinov2",
        "run_dir": "rtf_dinov2_128d/ae_dim128_meta_img",
        "group": "phase3",
    },
]


# ======================================================================
# MODEL LOADERS
# ======================================================================


def load_rtf(model_path, device, embed_dim=128, backbone="simple", d_model=256, in_channels=37, use_images=True):
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
            "latent_dim": embed_dim,
            "in_channels": in_channels,
            "d_model": d_model,
            "use_images": use_images,
            "image_backbone": backbone,
        }

    model = LightCurveCompressor(**model_kwargs)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()
    return model


def load_perceiver(model_path, device, embed_dim=128, backbone="simple", d_model=256, in_channels=37, use_images=True):
    """Load a Perceiver IO model with given embed_dim."""
    from perceiver_model import PerceiverAutoencoder

    model = PerceiverAutoencoder(
        phot_channels=7,
        meta_channels=30,
        d_model=d_model,
        n_latents=32,
        n_cross=2,
        n_self=6,
        n_heads=8,
        dropout=0.1,
        embed_dim=embed_dim,
        mask_ratio=0.50,
        image_backbone=backbone,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_astrodistill(model_path, device, embed_dim=128, backbone="simple", d_model=256, in_channels=37, use_images=True):
    """Load an AstroDistill model with given embed_dim and backbone."""
    from astrodistill_model import AstroDistill

    model = AstroDistill(
        phot_channels=7,
        meta_channels=30,
        d_model=d_model,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        embed_dim=embed_dim,
        mask_ratio=0.50,
        image_backbone=backbone,
        top_k_layers=2,
        pred_hidden=512,
        ema_decay=0.999,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


ARCH_LOADERS = {
    "rtf": load_rtf,
    "perceiver": load_perceiver,
    "astrodistill": load_astrodistill,
}


# ======================================================================
# EMBEDDING EXTRACTION
# ======================================================================


@torch.no_grad()
def extract_embeddings(model, fritz_dir, scores_path, device, in_channels=37):
    """Extract embeddings from Fritz NPZ files and merge labels.

    Returns dict with 'embeddings', 'labels_fine', 'obj_ids'.
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

            x_np = x_np[:, :in_channels]

            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            pad_mask = torch.zeros(1, x.shape[1], dtype=torch.bool).to(device)
            images = torch.from_numpy(image_np).unsqueeze(0).to(device)

            emb = model.embed(x, pad_mask, images)
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

    # Filter to top 5 most common classes (rare classes break CV and silhouette)
    top_5_classes = df_known["classification"].value_counts().nlargest(5).index
    is_top_5 = df_known["classification"].isin(top_5_classes)
    df_filtered = df_known[is_top_5]

    idx = df_filtered.index.values
    embeddings_arr = np.array(all_emb)[idx]
    labels_arr = df_filtered["classification"].values
    obj_ids_arr = df_filtered["obj_id"].values

    return {
        "embeddings": embeddings_arr,
        "labels_fine": labels_arr,
        "obj_ids": obj_ids_arr,
    }


# ======================================================================
# METRICS
# ======================================================================


def compute_metrics(embeddings, labels, encoder_name):
    """Compute clustering and classification metrics on embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    X = embeddings

    results = {
        "encoder": encoder_name,
        "n_samples": len(X),
        "n_classes": n_classes,
        "classes": list(le.classes_),
    }

    # Silhouette score (cosine distance)
    if n_classes >= 2:
        n_sil = min(len(X), 10000)
        idx = np.random.choice(len(X), n_sil, replace=False)
        results["silhouette"] = float(
            silhouette_score(X[idx], y[idx], metric="cosine")
        )
    else:
        results["silhouette"] = float("nan")

    # k-NN accuracy (k=10, stratified 5-fold CV)
    n_folds = min(5, min(np.bincount(y)))
    if n_classes >= 2 and n_folds >= 2:
        knn = KNeighborsClassifier(
            n_neighbors=min(10, len(X) // 5), metric="cosine",
        )
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        knn_scores = cross_val_score(knn, X, y, cv=skf, scoring="accuracy")
        results["knn_acc"] = float(knn_scores.mean())
        results["knn_std"] = float(knn_scores.std())
    else:
        results["knn_acc"] = float("nan")
        results["knn_std"] = float("nan")

    # Linear probe (logistic regression, stratified 5-fold CV)
    if n_classes >= 2 and n_folds >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(
            max_iter=2000, multi_class="multinomial", C=1.0, n_jobs=-1,
        )
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        lp_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
        results["linear_probe_acc"] = float(lp_scores.mean())
        results["linear_probe_std"] = float(lp_scores.std())
    else:
        results["linear_probe_acc"] = float("nan")
        results["linear_probe_std"] = float("nan")

    return results


def bootstrap_significance(emb_a, emb_b, labels, n_boot=1000):
    """Paired bootstrap test: is model A significantly better than B?

    Uses linear probe accuracy as the metric. Returns mean delta,
    95% confidence interval, and p-value. If CI excludes 0, the
    difference is statistically significant at α=0.05.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n = len(y)
    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        return None

    deltas = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        y_b = y[idx]

        # Skip if bootstrap sample has degenerate class distribution
        if len(np.unique(y_b)) < 2 or min(np.bincount(y_b)) < 2:
            continue

        n_folds_b = min(n_folds, min(np.bincount(y_b)))
        skf = StratifiedKFold(n_splits=n_folds_b, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0, n_jobs=-1)
        scaler = StandardScaler()

        X_a = scaler.fit_transform(emb_a[idx])
        score_a = cross_val_score(clf, X_a, y_b, cv=skf, scoring="accuracy").mean()

        X_b = scaler.fit_transform(emb_b[idx])
        score_b = cross_val_score(clf, X_b, y_b, cv=skf, scoring="accuracy").mean()

        deltas.append(score_a - score_b)

    if not deltas:
        return None

    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_value = float((deltas <= 0).mean())
    return {
        "mean_delta": float(deltas.mean()),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "p_value": p_value,
        "significant": bool(ci_lo > 0 or ci_hi < 0),
    }


def compute_isolation_forest_precision(embeddings, labels, encoder_name, k=20):
    """Train Isolation Forest on embeddings and measure precision@k.

    'Anomalous' is defined as any class with fewer than 10% representation
    in the dataset (rare classes). Precision@k = fraction of top-k most
    anomalous samples that are truly rare-class.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Define rare classes as those with < 10% representation
    counts = np.bincount(y)
    total = len(y)
    rare_mask = np.array([counts[yi] / total < 0.10 for yi in y])

    if rare_mask.sum() == 0 or rare_mask.sum() == total:
        return {"if_precision_at_k": float("nan"), "k": k, "n_rare": int(rare_mask.sum())}

    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(embeddings)
    scores = iso.score_samples(embeddings)  # more negative = more anomalous

    # Top-k most anomalous (lowest scores)
    top_k_idx = np.argsort(scores)[:k]
    precision = float(rare_mask[top_k_idx].mean())

    return {
        "if_precision_at_k": precision,
        "k": k,
        "n_rare": int(rare_mask.sum()),
        "rare_classes": [le.classes_[i] for i in range(len(counts)) if counts[i] / total < 0.10],
    }


def compute_per_class_f1(embeddings, labels, encoder_name):
    """Compute per-class F1 scores using stratified cross-val."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0, n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=skf)

    report = classification_report(
        y, y_pred, target_names=le.classes_, output_dict=True, zero_division=0,
    )
    return report


# ======================================================================
# PLOTTING
# ======================================================================


def plot_comparison_bar(results_list, output_dir):
    """Bar chart comparing all models on 3 metrics."""
    names = [r["encoder"] for r in results_list]
    sil = [r["silhouette"] for r in results_list]
    knn = [r["knn_acc"] for r in results_list]
    lp = [r["linear_probe_acc"] for r in results_list]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("10-Model Encoder Comparison", fontsize=16, fontweight="bold")

    x = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

    for ax, metric, title, vals in [
        (axes[0], sil, "Silhouette Score (cosine)", sil),
        (axes[1], knn, "k-NN Accuracy (k=10)", knn),
        (axes[2], lp, "Linear Probe Accuracy", lp),
    ]:
        bars = ax.barh(x, vals, color=colors)
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(max(vals) * 1.15, 0.01))
        # Annotate bars
        for bar, v in zip(bars, vals):
            ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "model_comparison_bars.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrices(report_a, report_b, name_a, name_b, output_dir):
    """Plot per-class F1 comparison between two models."""
    classes = [k for k in report_a if k not in ("accuracy", "macro avg", "weighted avg")]

    f1_a = [report_a[c]["f1-score"] for c in classes]
    f1_b = [report_b[c]["f1-score"] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width / 2, f1_a, width, label=name_a, color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1_b, width, label=name_b, color="#2ecc71", alpha=0.85)

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"Per-Class F1: {name_a} vs {name_b}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Annotate
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, "per_class_f1_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_umap_top4(encoder_data, output_dir):
    """UMAP 2x2 grid for the top 4 models."""
    try:
        from umap import UMAP  # noqa: F401
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        method = "t-SNE"

    n = min(4, len(encoder_data))

    # Consistent colormap
    all_labels = []
    for _, _, labels in encoder_data[:n]:
        all_labels.extend(labels.tolist())
    unique = sorted(set(all_labels))
    cmap = plt.cm.get_cmap("tab10", max(len(unique), 2))
    colormap = {cls: cmap(i) for i, cls in enumerate(unique)}

    rows = 2 if n > 2 else 1
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(14, 12))
    axes = np.array(axes).flatten()

    fig.suptitle(f"Embedding Space ({method}) — Top 4 Models",
                 fontsize=16, fontweight="bold", y=1.01)

    for i, (name, emb, labels) in enumerate(encoder_data[:n]):
        ax = axes[i]
        print(f"  Computing {method} for {name}...")

        n_plot = min(len(emb), 20000)
        idx = np.random.choice(len(emb), n_plot, replace=False)
        emb_sub = emb[idx]
        labels_sub = labels[idx]

        if method == "UMAP":
            from umap import UMAP as UMAPClass
            reducer = UMAPClass(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        else:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)

        coords = reducer.fit_transform(emb_sub)

        for cls in unique:
            mask = labels_sub == cls
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[colormap[cls]], s=8, alpha=0.6)

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colormap[cls],
               markersize=8, label=cls)
        for cls in unique
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(unique), 8), fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    path = os.path.join(output_dir, "umap_top4.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# MAIN
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare all 13 encoder variants (Phase 1 + Phase 2)"
    )
    parser.add_argument("--fritz-dir", required=True,
                        help="Path to Fritz sources NPZ dir")
    parser.add_argument("--scores", required=True,
                        help="Path to anomaly_scores.csv with classifications")
    parser.add_argument("--runs-dir", required=True,
                        help="Root runs directory containing all model subdirs")
    parser.add_argument("--output-dir", default="runs/comparison_v2",
                        help="Output directory for results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(42)

    print("=" * 70)
    print("  ENCODER COMPARISON v2: 10-Model Grid Evaluation")
    print("=" * 70)

    # --- Discover available models ---
    available = []
    for spec in MODEL_REGISTRY:
        model_path = os.path.join(args.runs_dir, spec["run_dir"], "best_model.pt")
        if os.path.exists(model_path):
            spec["model_path"] = model_path
            available.append(spec)
            print(f"  ✓ {spec['name']:35s}  {model_path}")
        else:
            print(f"  ✗ {spec['name']:35s}  NOT FOUND (skipping)")

    if not available:
        print("ERROR: No models found. Check --runs-dir path.")
        return

    print(f"\n  Found {len(available)} / {len(MODEL_REGISTRY)} models.\n")

    # --- Extract embeddings and compute metrics for each ---
    all_results = []
    all_emb_data = {}  # name -> emb_data dict
    all_reports = {}   # name -> per-class F1 report

    for i, spec in enumerate(available):
        name = spec["name"]
        print(f"\n[{i+1}/{len(available)}] {name}")
        print(f"  arch={spec['arch']}, embed_dim={spec['embed_dim']}, "
              f"backbone={spec['backbone']}")

        # Load
        loader = ARCH_LOADERS[spec["arch"]]
        t0 = time.time()
        model = loader(
            spec["model_path"], args.device,
            embed_dim=spec["embed_dim"],
            backbone=spec["backbone"],
            d_model=spec.get("d_model", 256),
            in_channels=spec.get("in_channels", 37),
            use_images=spec.get("use_images", True),
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded: {n_params:,} params ({time.time() - t0:.1f}s)")

        # Extract embeddings
        t0 = time.time()
        emb_data = extract_embeddings(
            model, args.fritz_dir, args.scores, args.device,
            in_channels=spec.get("in_channels", 37)
        )
        emb = emb_data["embeddings"]
        print(f"  Extracted: {emb.shape} in {time.time() - t0:.1f}s")

        # Save embeddings
        safe_name = name.lower().replace(" ", "_").replace("+", "_")
        emb_path = os.path.join(args.output_dir, f"emb_{safe_name}.npz")
        np.savez_compressed(emb_path, **emb_data)

        # Metrics
        metrics = compute_metrics(emb, emb_data["labels_fine"], name)
        metrics["n_params"] = n_params
        metrics["embed_dim"] = spec["embed_dim"]
        metrics["backbone"] = spec["backbone"]
        metrics["group"] = spec["group"]
        all_results.append(metrics)

        print(f"    Silhouette:    {metrics['silhouette']:.4f}")
        print(f"    k-NN (k=10):   {metrics['knn_acc']:.4f} ± {metrics['knn_std']:.4f}")
        print(f"    Linear Probe:  {metrics['linear_probe_acc']:.4f} ± "
              f"{metrics['linear_probe_std']:.4f}")

        # Isolation Forest anomaly detection precision
        if_metrics = compute_isolation_forest_precision(emb, emb_data["labels_fine"], name)
        metrics.update(if_metrics)
        if not np.isnan(if_metrics["if_precision_at_k"]):
            print(f"    IF Prec@{if_metrics['k']}:  {if_metrics['if_precision_at_k']:.4f} "
                  f"(rare classes: {if_metrics.get('rare_classes', [])})")

        # Per-class F1
        report = compute_per_class_f1(emb, emb_data["labels_fine"], name)
        if report:
            all_reports[name] = report

        all_emb_data[name] = emb_data
        del model
        torch.cuda.empty_cache()

    # --- Print comparison table ---
    print(f"\n{'=' * 100}")
    print("FULL COMPARISON TABLE")
    print(f"{'=' * 100}")
    print(
        f"{'Model':<35} {'Dim':>4} {'Backbone':<8} {'Params':>10} "
        f"{'Silhouette':>11} {'k-NN':>8} {'LinProbe':>9}"
    )
    print("-" * 95)

    # Sort by linear probe accuracy
    sorted_results = sorted(all_results, key=lambda r: r.get("linear_probe_acc", -1), reverse=True)
    for r in sorted_results:
        print(
            f"{r['encoder']:<35} {r['embed_dim']:>4} {r['backbone']:<8} "
            f"{r['n_params']:>10,} "
            f"{r['silhouette']:>11.4f} "
            f"{r['knn_acc']:>8.4f} "
            f"{r['linear_probe_acc']:>9.4f}"
        )
    print(f"{'=' * 100}")

    # --- Determine winners ---
    best_sil = max(all_results, key=lambda r: r.get("silhouette", -1))
    best_knn = max(all_results, key=lambda r: r.get("knn_acc", -1))
    best_lp = max(all_results, key=lambda r: r.get("linear_probe_acc", -1))

    print(f"\n  🏆 Best Silhouette:  {best_sil['encoder']} ({best_sil['silhouette']:.4f})")
    print(f"  🏆 Best k-NN:        {best_knn['encoder']} ({best_knn['knn_acc']:.4f})")
    print(f"  🏆 Best Linear Probe:{best_lp['encoder']} ({best_lp['linear_probe_acc']:.4f})")

    # --- Bar chart ---
    print("\nGenerating comparison bar chart...")
    plot_comparison_bar(sorted_results, args.output_dir)

    # --- Head-to-head: RTF baseline vs overall winner ---
    winner = best_lp  # Use linear probe as the definitive metric
    rtf_result = next((r for r in all_results if r["encoder"] == "RTF 128-D"), None)

    if rtf_result and winner["encoder"] != "RTF 128-D":
        print(f"\n{'=' * 70}")
        print(f"HEAD-TO-HEAD: RTF 128-D vs {winner['encoder']}")
        print(f"{'=' * 70}")
        print(f"{'Metric':<25} {'RTF 128-D':>15} {winner['encoder']:>25} {'Δ':>10}")
        print("-" * 75)

        for metric, label in [
            ("silhouette", "Silhouette"),
            ("knn_acc", "k-NN Accuracy"),
            ("linear_probe_acc", "Linear Probe"),
        ]:
            rv = rtf_result.get(metric, 0)
            wv = winner.get(metric, 0)
            delta = wv - rv
            sign = "+" if delta > 0 else ""
            print(f"{label:<25} {rv:>15.4f} {wv:>25.4f} {sign}{delta:>9.4f}")

        print(f"{'Parameters':<25} {rtf_result['n_params']:>15,} "
              f"{winner['n_params']:>25,}")
        print(f"{'Embed Dim':<25} {rtf_result.get('embed_dim', 128):>15} "
              f"{winner.get('embed_dim', '?'):>25}")
        print(f"{'Backbone':<25} {'simple':>15} {winner.get('backbone', '?'):>25}")
        print(f"{'=' * 70}")

        # Per-class F1 comparison plot
        rtf_name = "RTF 128-D"
        winner_name = winner["encoder"]
        if rtf_name in all_reports and winner_name in all_reports:
            print("\nGenerating per-class F1 comparison...")
            plot_confusion_matrices(
                all_reports[rtf_name], all_reports[winner_name],
                rtf_name, winner_name, args.output_dir,
            )

    # --- UMAP for top 4 ---
    top4_names = [r["encoder"] for r in sorted_results[:4]]
    umap_data = [
        (name, all_emb_data[name]["embeddings"], all_emb_data[name]["labels_fine"])
        for name in top4_names
        if name in all_emb_data
    ]
    if umap_data:
        print("\nGenerating UMAP for top 4 models...")
        plot_umap_top4(umap_data, args.output_dir)

    # --- Bootstrap significance test: winner vs RTF 128-D baseline ---
    if rtf_result and winner["encoder"] != "RTF 128-D":
        winner_name = winner["encoder"]
        if winner_name in all_emb_data and "RTF 128-D" in all_emb_data:
            print(f"\nRunning paired bootstrap significance test "
                  f"({winner_name} vs RTF 128-D, n=1000)...")
            sig = bootstrap_significance(
                all_emb_data[winner_name]["embeddings"],
                all_emb_data["RTF 128-D"]["embeddings"],
                all_emb_data[winner_name]["labels_fine"],
                n_boot=1000,
            )
            if sig:
                sig_str = "YES ✓" if sig["significant"] else "NO ✗"
                print(f"  Mean Δ (linear probe):  {sig['mean_delta']:+.4f}")
                print(f"  95% CI:                 [{sig['ci_95_lo']:+.4f}, {sig['ci_95_hi']:+.4f}]")
                print(f"  p-value:                {sig['p_value']:.4f}")
                print(f"  Statistically significant: {sig_str}")

    # --- Generate publication-quality markdown summary ---
    md_lines = [
        "# Encoder Comparison Results",
        "",
        "**Date**: generated by compare_encoders_v2.py",
        f"**Models evaluated**: {len(sorted_results)}",
        f"**Test sources**: {sorted_results[0]['n_samples']} labeled Fritz sources",
        f"**Classes**: {sorted_results[0]['n_classes']} ({', '.join(sorted_results[0].get('classes', []))})",
        "",
        "## Rankings (sorted by Linear Probe accuracy)",
        "",
        "| Rank | Model | Dim | Backbone | Params | Silhouette | k-NN | LinProbe | IF P@20 |",
        "|------|-------|-----|----------|--------|------------|------|----------|---------|",
    ]
    for rank, r in enumerate(sorted_results, 1):
        if_p = r.get("if_precision_at_k", float("nan"))
        if_str = f"{if_p:.3f}" if not np.isnan(if_p) else "N/A"
        md_lines.append(
            f"| {rank} | {r['encoder']} | {r.get('embed_dim', '?')} | "
            f"{r.get('backbone', '?')} | {r.get('n_params', 0):,} | "
            f"{r['silhouette']:.4f} | {r['knn_acc']:.4f} | "
            f"{r['linear_probe_acc']:.4f} | {if_str} |"
        )

    # Add significance test result if available
    if rtf_result and winner["encoder"] != "RTF 128-D" and sig:
        md_lines.extend([
            "",
            "## Statistical Significance",
            "",
            f"**{winner['encoder']}** vs **RTF 128-D** (paired bootstrap, n=1000):",
            "",
            f"- Mean Δ (linear probe): `{sig['mean_delta']:+.4f}`",
            f"- 95% CI: `[{sig['ci_95_lo']:+.4f}, {sig['ci_95_hi']:+.4f}]`",
            f"- p-value: `{sig['p_value']:.4f}`",
            f"- **Statistically significant**: {'YES ✓' if sig['significant'] else 'NO ✗'}",
        ])

    md_lines.extend(["", ""])
    md_path = os.path.join(args.output_dir, "comparison_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\n  Markdown summary saved: {md_path}")

    # --- Save JSON ---
    results_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(sorted_results, f, indent=2)
    print(f"  Results saved: {results_path}")

    print(f"\n{'=' * 70}")
    print("  COMPARISON COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
