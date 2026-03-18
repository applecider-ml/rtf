"""
Latent space visualization: UMAP and t-SNE projections of embeddings.

Generates publication-quality plots colored by transient class, showing how
the latent space organizes by astrophysical type at different compression levels.

Usage:
    python src/visualize.py \
        --runs-dir runs \
        --output-dir analysis/visualizations \
        --models ae_dim64 ae_dim256
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

COARSE_NAMES = {0: "SN Ia", 1: "SN cc", 2: "Cataclysmic", 3: "AGN", 4: "TDE"}
FINE_NAMES = {
    0: "SN Ia",
    1: "SN Ib",
    2: "SN Ic",
    3: "SN II",
    4: "SN IIP",
    5: "SN IIn",
    6: "SN IIb",
    7: "Cataclysmic",
    8: "AGN",
    9: "TDE",
}

# Colorblind-friendly palette
COARSE_COLORS = {
    0: "#0072B2",  # SN Ia — blue
    1: "#D55E00",  # SN cc — orange
    2: "#009E73",  # Cataclysmic — green
    3: "#CC79A7",  # AGN — pink
    4: "#E69F00",  # TDE — gold
}

FINE_COLORS = {
    0: "#0072B2",  # SN Ia
    1: "#D55E00",  # SN Ib
    2: "#E69F00",  # SN Ic
    3: "#56B4E9",  # SN II
    4: "#009E73",  # SN IIP
    5: "#F0E442",  # SN IIn
    6: "#CC79A7",  # SN IIb
    7: "#000000",  # Cataclysmic
    8: "#999999",  # AGN
    9: "#FF0000",  # TDE
}


def project_2d(embeddings, method="umap", **kwargs):
    """Project embeddings to 2D using UMAP or t-SNE.

    Returns (projection, display_name, method_key) where method_key
    is the actual method used (may differ from input if fallback occurs).
    """
    if method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            print("  umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "umap":
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=kwargs.get("n_neighbors", 30),
            min_dist=kwargs.get("min_dist", 0.3),
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings), "UMAP", "umap"

    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=kwargs.get("perplexity", 30),
            random_state=42,
            init="pca",
        )
        return reducer.fit_transform(embeddings), "t-SNE", "tsne"

    elif method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(embeddings), "PCA", "pca"


def plot_latent_scatter(
    proj,
    labels,
    names,
    colors,
    title,
    output_path,
    point_size=3,
    alpha=0.5,
    figsize=(10, 8),
):
    """Scatter plot of 2D projected embeddings colored by class."""
    fig, ax = plt.subplots(figsize=figsize)

    # Plot rare classes last (on top) for visibility
    unique_labels = np.unique(labels)
    counts = {lbl: (labels == lbl).sum() for lbl in unique_labels}
    sorted_labels = sorted(unique_labels, key=lambda lbl: counts[lbl], reverse=True)

    for lbl in sorted_labels:
        mask = labels == lbl
        name = names.get(lbl, str(lbl))
        color = colors.get(lbl, "#333333")
        n = mask.sum()
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=point_size,
            alpha=alpha,
            c=color,
            label=f"{name} (n={n})",
            rasterized=True,
        )

    ax.legend(markerscale=5, fontsize=9, loc="best", framealpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_recon_error_map(proj, recon_errors, title, output_path, figsize=(10, 8)):
    """Scatter plot colored by reconstruction error (anomaly score)."""
    fig, ax = plt.subplots(figsize=figsize)

    # Clip outliers for better colormap
    vmax = np.percentile(recon_errors, 95)
    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        s=3,
        c=recon_errors,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        alpha=0.6,
        rasterized=True,
    )
    plt.colorbar(scatter, ax=ax, label="Reconstruction error", shrink=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_multi_dim_comparison(
    all_projs,
    all_labels,
    names,
    colors,
    dims,
    method_name,
    output_path,
    figsize=(20, 5),
):
    """Side-by-side latent space plots at different compression levels."""
    n = len(dims)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    unique_labels = np.unique(all_labels[0])
    counts = {lbl: (all_labels[0] == lbl).sum() for lbl in unique_labels}
    sorted_labels = sorted(unique_labels, key=lambda lbl: counts[lbl], reverse=True)

    for i, (proj, labels, dim) in enumerate(zip(all_projs, all_labels, dims)):
        ax = axes[i]
        for lbl in sorted_labels:
            mask = labels == lbl
            name = names.get(lbl, str(lbl))
            color = colors.get(lbl, "#333333")
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                s=2,
                alpha=0.4,
                c=color,
                label=name if i == 0 else None,
                rasterized=True,
            )
        ax.set_title(f"dim={dim}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(markerscale=5, fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(f"Latent space structure ({method_name})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def visualize_run(run_path, output_dir, methods=("umap", "tsne")):
    """Generate all visualizations for a single model run."""
    emb_path = os.path.join(run_path, "test_embeddings.npz")
    summary_path = os.path.join(run_path, "summary.json")

    if not os.path.exists(emb_path):
        return

    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels_coarse = data["labels_coarse"]
    labels_fine = data["labels_fine"]
    recon_errors = data["recon_errors"]

    with open(summary_path) as f:
        summary = json.load(f)

    run_name = os.path.basename(run_path)
    mode = summary["mode"]
    dim = summary["latent_dim"]
    out = os.path.join(output_dir, run_name)
    os.makedirs(out, exist_ok=True)

    print(f"\nVisualizing {run_name} ({mode}, dim={dim}, {len(embeddings)} samples)")

    for method in methods:
        proj, method_name, method_key = project_2d(embeddings, method=method)

        # Coarse classes
        plot_latent_scatter(
            proj,
            labels_coarse,
            COARSE_NAMES,
            COARSE_COLORS,
            f"{mode.upper()} dim={dim} — Coarse classes ({method_name})",
            os.path.join(out, f"coarse_{method_key}.png"),
        )

        # Fine classes
        plot_latent_scatter(
            proj,
            labels_fine,
            FINE_NAMES,
            FINE_COLORS,
            f"{mode.upper()} dim={dim} — Fine classes ({method_name})",
            os.path.join(out, f"fine_{method_key}.png"),
        )

        # Reconstruction error heatmap
        plot_recon_error_map(
            proj,
            recon_errors,
            f"{mode.upper()} dim={dim} — Reconstruction error ({method_name})",
            os.path.join(out, f"recon_error_{method_key}.png"),
        )


def main():
    parser = argparse.ArgumentParser(description="Latent space visualization")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific run dirs to visualize (default: all)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["umap", "tsne"],
        choices=["umap", "tsne", "pca"],
    )
    parser.add_argument(
        "--comparison-dims",
        nargs="*",
        type=int,
        default=None,
        help="Dims for side-by-side comparison plot (e.g. 8 32 128 512)",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only generate comparison plots, skip individual run visualizations",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_dirs = args.models or sorted(
        [
            d
            for d in os.listdir(args.runs_dir)
            if os.path.isdir(os.path.join(args.runs_dir, d))
        ]
    )

    # Individual run visualizations
    if not args.comparison_only:
        for d in run_dirs:
            run_path = os.path.join(args.runs_dir, d)
            visualize_run(run_path, args.output_dir, methods=args.methods)

    # Multi-dim comparison plot
    if args.comparison_dims:
        for method in args.methods:
            all_projs, all_labels, dims_found = [], [], []
            for dim in args.comparison_dims:
                # Find the AE run for this dim
                for d in run_dirs:
                    if f"ae_dim{dim}" == d or d == f"ae_dim{dim}_meta":
                        emb_path = os.path.join(args.runs_dir, d, "test_embeddings.npz")
                        if os.path.exists(emb_path):
                            data = np.load(emb_path, allow_pickle=True)
                            proj, method_name, method_key = project_2d(
                                data["embeddings"], method=method
                            )
                            all_projs.append(proj)
                            all_labels.append(data["labels_coarse"])
                            dims_found.append(dim)
                        break

            if len(all_projs) >= 2:
                plot_multi_dim_comparison(
                    all_projs,
                    all_labels,
                    COARSE_NAMES,
                    COARSE_COLORS,
                    dims_found,
                    method_name,
                    os.path.join(args.output_dir, f"comparison_{method_key}.png"),
                )


if __name__ == "__main__":
    main()
