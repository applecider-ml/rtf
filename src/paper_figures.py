"""
Paper Figures: Self-supervised Anomaly Detection in ZTF Alerts.
================================================================
Reads pre-computed comparison_results.json and embedding .npz files,
produces publication-quality figures for the IEEE paper.

All figures share:
  - consistent color palette keyed by architecture
  - consistent marker/hatch keyed by backbone
  - same class->color mapping across every scatter/UMAP
  - PDF + PNG output at 300 DPI
  - no figure has a bare line without error bars or spread

Usage:
    python paper_figures.py \
        --results /path/to/comparison_results.json \
        --emb-dir /path/to/comparison_v3/ \
        --output-dir /path/to/paper/figures


"""

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ======================================================================
# DESIGN SYSTEM — every figure in this paper uses these constants.
# Color means the same thing in figure 1 and figure 6.
# ======================================================================

# Architecture colors (colorblind-safe)
ARCH_COLORS = {
    "RTF":          "#2563EB",   # blue
    "Perceiver":    "#DC2626",   # red
    "AstroDistill": "#059669",   # teal-green
}

# Backbone hatching and edge style
BACKBONE_HATCH = {
    "simple":       "",          # solid fill
    "dinov2":       "///",       # diagonal
    "zoobot":       "xxx",       # cross
    "zoobot_large": "\\\\\\",   # reverse diagonal
}

BACKBONE_LABELS = {
    "simple":       "Simple CNN",
    "dinov2":       "DINOv2",
    "zoobot":       "Zoobot Pico",
    "zoobot_large": "Zoobot Large",
}

# Transient class colors (consistent across all scatter/UMAP plots)
CLASS_COLORS = {
    "Ia":                       "#3B82F6",  # blue
    "Type II":                  "#EF4444",  # red
    "IIn":                      "#F59E0B",  # amber
    "Ic-SLSN":                  "#8B5CF6",  # purple
    "Tidal Disruption Event":   "#10B981",  # emerald
}

CLASS_SHORT = {
    "Ia":                       "SN Ia",
    "Type II":                  "SN II",
    "IIn":                      "SN IIn",
    "Ic-SLSN":                  "SLSN-Ic",
    "Tidal Disruption Event":   "TDE",
}

# Matplotlib RC overrides for the whole script
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
})

# Chance-level baselines (5-class uniform)
# RULE: every figure has the null result drawn on it.
CHANCE_LEVELS = {
    "knn_acc":            1.0 / 5,          # 0.200 — random 5-class
    "linear_probe_acc":   1.0 / 5,          # 0.200
    "if_precision_at_k":  22.0 / 414,       # 0.053 — TDE base rate
}
CHANCE_LABEL = "chance (1/5)" 
IF_CHANCE_LABEL = "TDE base rate"


# ======================================================================
# HELPER: draw chance-level baseline on an axis
# ======================================================================

def draw_chance_line(ax, metric_key, orientation="vertical"):
    """Draw a dashed chance-level line if this metric has one."""
    if metric_key not in CHANCE_LEVELS:
        return
    level = CHANCE_LEVELS[metric_key]
    label = IF_CHANCE_LABEL if "if_" in metric_key else CHANCE_LABEL
    kwargs = {"color": "gray", "linestyle": "--", "linewidth": 1, "alpha": 0.6,
                  "label": label, "zorder": 1}
    if orientation == "horizontal":
        ax.axvline(level, **kwargs)
    else:
        ax.axhline(level, **kwargs)


# ======================================================================
# HELPER: parse arch name from encoder string
# ======================================================================

def parse_arch(encoder_name):
    """Return base architecture key from encoder name string."""
    if encoder_name.startswith("RTF"):
        return "RTF"
    elif encoder_name.startswith("Perceiver"):
        return "Perceiver"
    elif encoder_name.startswith("AstroDistill"):
        return "AstroDistill"
    return "Unknown"


def format_params(n):
    """Format parameter count: 3,617,498 -> '3.6M'."""
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(n)


def save_fig(fig, output_dir, name):
    """Save as both PDF (vector) and PNG (raster)."""
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# ======================================================================
# FIGURE 1: Full 16-model leaderboard — horizontal bars, 4 metrics
# ======================================================================

def fig_leaderboard(results, output_dir):
    """
    All 16 models ranked by linear probe accuracy.
    Four subplots: Silhouette, k-NN, Linear Probe, IF P@20.
    Color = architecture, hatch = backbone.
    RULE: every bar has error bars where available.
    """
    # Sort by linear probe
    results = sorted(results, key=lambda r: r["linear_probe_acc"])

    names = [r["encoder"] for r in results]
    n = len(names)
    y = np.arange(n)

    metrics = [
        ("linear_probe_acc",   "Linear Probe Accuracy",  "linear_probe_std", None),
        ("knn_acc",            "$k$-NN Accuracy",        "knn_std",      None),
        ("if_precision_at_k",  "IF Precision @ 20",      None,           None),
        ("silhouette",         "Silhouette Score",       None,           None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)

    for ax, (key, title, std_key, _) in zip(axes, metrics):
        vals = [r.get(key, 0) for r in results]
        errs = [r.get(std_key, 0) if std_key else 0 for r in results]

        colors = [ARCH_COLORS[parse_arch(r["encoder"])] for r in results]
        hatches = [BACKBONE_HATCH.get(r.get("backbone", "simple"), "") for r in results]

        bars = ax.barh(y, vals, xerr=errs if std_key else None,
                       color=colors, edgecolor="black", linewidth=0.5,
                       capsize=2, height=0.7)

        # Apply hatches
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Chance-level baseline
        draw_chance_line(ax, key, orientation="horizontal")

        ax.set_title(title, fontweight="bold")
        ax.set_yticks(y)
        if ax == axes[0]:
            # Labels include param count on second line to avoid overlap
            labels = [f"{r['encoder']}\n({format_params(r['n_params'])})" for r in results]
            ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_yticklabels([])

        # Value annotations
        for i, (bar, v) in enumerate(zip(bars, vals)):
            x_pos = max(v, 0) + 0.005
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=6, color="black")

    # Build legend: architecture colors + backbone hatches
    arch_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=a)
                    for a, c in ARCH_COLORS.items()]
    bb_handles = [Patch(facecolor="white", edgecolor="black", linewidth=0.5,
                        hatch=h, label=BACKBONE_LABELS[b])
                  for b, h in BACKBONE_HATCH.items()]

    fig.legend(handles=arch_handles + bb_handles,
               loc="lower center", ncol=7, fontsize=8,
               bbox_to_anchor=(0.5, -0.06), frameon=True)

    fig.suptitle("16-Model Comparison: Self-Supervised Encoders for ZTF Alert Anomaly Detection",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    save_fig(fig, output_dir, "fig_leaderboard")


# ======================================================================
# FIGURE 2: Dimensionality ablation — grouped bars (128/64/32)
# ======================================================================

def fig_dimensionality(results, output_dir):
    """
    Grouped bar chart: 128-D vs 64-D vs 32-D for each baseline architecture.
    Only simple-backbone models.
    Shows Linear Probe and IF P@20 side by side.
    RULE: highlight the 64-D anomaly detection improvement.
    """
    archs = ["RTF", "Perceiver", "AstroDistill"]
    dims = [128, 64, 32]
    dim_colors = {128: "#3B82F6", 64: "#F59E0B", 32: "#EF4444"}

    # Filter to simple backbone only
    simple = [r for r in results if r.get("backbone") == "simple"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (metric, title, std_key) in zip(axes, [
        ("linear_probe_acc", "Linear Probe Accuracy", "linear_probe_std"),
        ("if_precision_at_k", "IF Precision @ 20", None),
    ]):
        x = np.arange(len(archs))
        width = 0.25

        for i, dim in enumerate(dims):
            vals = []
            errs = []
            for arch in archs:
                match = [r for r in simple
                         if parse_arch(r["encoder"]) == arch and r["embed_dim"] == dim]
                if match:
                    vals.append(match[0].get(metric, 0))
                    errs.append(match[0].get(std_key, 0) if std_key else 0)
                else:
                    vals.append(0)
                    errs.append(0)

            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width,
                          yerr=errs if std_key else None,
                          label=f"{dim}-D", color=dim_colors[dim],
                          edgecolor="black", linewidth=0.5, capsize=3)

            # Annotate values
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        # Chance-level baseline
        draw_chance_line(ax, metric)

        ax.set_xticks(x)
        ax.set_xticklabels(archs, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(title)
        ax.legend(title="Latent Dim", frameon=True)

    fig.suptitle("Effect of Embedding Dimensionality (Simple CNN Backbone)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, output_dir, "fig_dimensionality")


# ======================================================================
# FIGURE 3: Backbone impact — RTF with vs without pretrained backbones
# ======================================================================

def fig_backbone_impact(results, output_dir):
    """
    Grouped comparison: RTF baseline vs RTF+DINOv2 vs RTF+ZooP vs RTF+ZooL.
    Shows that pretrained backbones HURT RTF in unsupervised regime.
    RULE: this is the key finding — pretrained backbones hurt RTF in the unsupervised regime.
    """
    # All RTF variants at 128-D
    rtf_128 = [r for r in results
               if parse_arch(r["encoder"]) == "RTF" and r["embed_dim"] == 128]
    rtf_128 = sorted(rtf_128, key=lambda r: r["linear_probe_acc"], reverse=True)

    if not rtf_128:
        print("  SKIP: No RTF 128-D models found for backbone comparison.")
        return

    names = [r["encoder"] for r in rtf_128]
    metrics = [
        ("linear_probe_acc", "Linear Probe", "linear_probe_std"),
        ("knn_acc", "$k$-NN Accuracy", "knn_std"),
        ("if_precision_at_k", "IF P@20", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(names))

    for ax, (key, title, std_key) in zip(axes, metrics):
        vals = [r.get(key, 0) for r in rtf_128]
        errs = [r.get(std_key, 0) if std_key else 0 for r in rtf_128]
        [BACKBONE_HATCH.get(r.get("backbone", "simple"), "")
                  for r in rtf_128]

        bars = ax.bar(x, vals, yerr=errs if std_key else None,
                      color=ARCH_COLORS["RTF"], edgecolor="black",
                      linewidth=0.5, capsize=3, width=0.6)

        # Apply backbone hatches
        for bar, r in zip(bars, rtf_128):
            bar.set_hatch(BACKBONE_HATCH.get(r.get("backbone", "simple"), ""))

        ax.set_xticks(x)
        ax.set_xticklabels([BACKBONE_LABELS.get(r.get("backbone", "simple"),
                           r.get("backbone", "?")) for r in rtf_128],
                           rotation=0, ha="center", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(title)

        # Value annotations
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # Chance-level baseline
        draw_chance_line(ax, key)

    # Params as second line of x-tick label (avoids overlap with bar annotations)
    tick_labels = [
        f"{BACKBONE_LABELS.get(r.get('backbone', 'simple'), r.get('backbone', '?'))}\n({format_params(r['n_params'])})"
        for r in rtf_128
    ]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=8, ha="center")

    fig.suptitle("RTF Performance vs Vision Backbone (128-D) — Pretrained Backbones Hurt",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, output_dir, "fig_backbone_impact")


# ======================================================================
# FIGURE 4: UMAP embeddings — top 4 models, 2x2 grid
# ======================================================================

def fig_umap(results, emb_dir, output_dir):
    """
    2x2 UMAP grid of the top 4 models by linear probe accuracy.
    Consistent class colors across all 4 panels.
    RULE: same color = same class in every figure.
    """
    try:
        from umap import UMAP
    except ImportError:
        print("  SKIP: umap-learn not installed. Install with: pip install umap-learn")
        return

    # Top 4 by linear probe
    ranked = sorted(results, key=lambda r: r["linear_probe_acc"], reverse=True)[:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, r in enumerate(ranked):
        name = r["encoder"]
        safe_name = name.lower().replace(" ", "_").replace("+", "_")
        emb_path = os.path.join(emb_dir, f"emb_{safe_name}.npz")

        if not os.path.exists(emb_path):
            # Try alternate naming
            dim = r.get("embed_dim", 128)
            safe_name.replace(f"_{dim}-d", f"_{dim}-d")
            emb_path = os.path.join(emb_dir, f"emb_{safe_name}.npz")

        if not os.path.exists(emb_path):
            print(f"  WARN: embedding not found for {name} at {emb_path}")
            axes[i].text(0.5, 0.5, f"Missing: {name}", transform=axes[i].transAxes,
                         ha="center", va="center")
            continue

        data = np.load(emb_path, allow_pickle=True)
        emb = data["embeddings"]
        labels = data["labels_fine"]

        print(f"  Computing UMAP for {name} ({emb.shape})...")
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                       random_state=42, metric="cosine")
        coords = reducer.fit_transform(emb)

        ax = axes[i]
        for cls, cls_color in CLASS_COLORS.items():
            mask = labels == cls
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=cls_color, s=12, alpha=0.7,
                       edgecolors="none", label=CLASS_SHORT.get(cls, cls))

        ax.set_title(f"{name}\n({format_params(r['n_params'])} params, "
                     f"LP={r['linear_probe_acc']:.3f})",
                     fontweight="bold", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    legend_handles = [Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=c, markersize=8,
                             label=CLASS_SHORT.get(cls, cls))
                      for cls, c in CLASS_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle("UMAP Projections — Top 4 Models by Linear Probe Accuracy",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, output_dir, "fig_umap_top4")


# ======================================================================
# FIGURE 5: Params vs Performance — scatter (Dr. KM addition)
# Efficiency argument in one glance: RTF does more with less.
# ======================================================================

def fig_params_vs_performance(results, output_dir):
    """
    Scatter plot: x = log(params), y = linear probe accuracy.
    Bubble size = IF P@20. Color = architecture.
    RULE: this is the figure that makes the efficiency argument
    without a single sentence of prose. The reader sees it instantly.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        arch = parse_arch(r["encoder"])
        color = ARCH_COLORS.get(arch, "gray")
        backbone = r.get("backbone", "simple")
        BACKBONE_HATCH.get(backbone, "")
        params = r["n_params"]
        lp = r["linear_probe_acc"]
        ifp = r.get("if_precision_at_k", 0)

        # Bubble size proportional to IF P@20
        size = max(ifp * 500, 40)

        ax.scatter(params, lp, s=size, c=color, alpha=0.7,
                   edgecolors="black", linewidth=0.5, zorder=3)

        # Annotate with short name
        dim = r.get("embed_dim", "?")
        short = f"{dim}D"
        if backbone != "simple":
            short += f"+{BACKBONE_LABELS.get(backbone, backbone)[:4]}"
        ax.annotate(short, (params, lp), fontsize=5.5,
                    textcoords="offset points", xytext=(5, 3),
                    color="black", alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)", fontweight="bold")
    ax.set_ylabel("Linear Probe Accuracy", fontweight="bold")
    ax.set_title("Parameter Efficiency: More Parameters ≠ Better Anomaly Detection",
                 fontweight="bold")

    # Format x-axis
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: format_params(int(x))))

    # Legend: architecture color
    arch_handles = [Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, markersize=8, label=a,
                           markeredgecolor="black", markeredgewidth=0.5)
                    for a, c in ARCH_COLORS.items()]

    # Legend: bubble size
    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=np.sqrt(s/np.pi), label=f"IF P@20 = {v:.2f}",
               markeredgecolor="black", markeredgewidth=0.5)
        for v, s in [(0.00, 40), (0.10, max(0.10*500, 40)), (0.20, 0.20*500)]
    ]

    ax.legend(handles=arch_handles + size_handles, loc="lower left",
              fontsize=7, frameon=True, ncol=2)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig_params_vs_performance")


# ======================================================================
# FIGURE 6: Radar chart — top 3 baselines across all 4 metrics
# Shows the trade-off shape, not just individual numbers.
# ======================================================================

def fig_radar(results, output_dir):
    """
    Radar/spider chart comparing the 3 baseline architectures (128-D, simple CNN)
    across all 4 evaluation metrics.
    RULE: shows the shape of the trade-off, not just bar heights.
    """
    baselines = [r for r in results
                 if r.get("backbone") == "simple" and r.get("embed_dim") == 128]

    if len(baselines) < 3:
        print("  SKIP: Not enough 128-D baseline models for radar chart.")
        return

    metrics = ["silhouette", "knn_acc", "linear_probe_acc", "if_precision_at_k"]
    labels = ["Silhouette", "$k$-NN", "Linear Probe", "IF P@20"]
    n_metrics = len(metrics)

    # Normalize each metric to [0, 1] for radar display
    all_vals = {m: [r.get(m, 0) for r in results] for m in metrics}
    mins = {m: min(v) for m, v in all_vals.items()}
    maxs = {m: max(v) for m, v in all_vals.items()}

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for r in baselines:
        arch = parse_arch(r["encoder"])
        color = ARCH_COLORS.get(arch, "gray")

        values = []
        for m in metrics:
            v = r.get(m, 0)
            rng = maxs[m] - mins[m]
            if rng > 0:
                values.append((v - mins[m]) / rng)
            else:
                values.append(0.5)
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=r["encoder"],
                color=color, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 1.1)
    # Add normalized note so reader knows axes are min-max scaled across all 16 models
    ax.set_title("Baseline Architecture Comparison (128-D, Simple CNN)\n"
                 "All axes normalized 0\u20131 across all 16 models",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)

    fig.tight_layout()
    save_fig(fig, output_dir, "fig_radar_baselines")


# ======================================================================
# FIGURE 7: Architecture × Backbone heatmap
# Inspired by Nabeel's clean tabular presentation, but as a visual.
# Makes the interaction effect undeniable at a glance.
# ======================================================================

def fig_heatmap(results, output_dir):
    """
    Heatmap: rows = architectures, cols = backbones.
    Cell value = linear probe accuracy. Color intensity = performance.
    RULE: this figure makes the "pretrained hurts RTF but helps
    AstroDistill" finding visually undeniable in one glance.
    """
    archs = ["RTF", "Perceiver", "AstroDistill"]
    backbones_order = ["simple", "dinov2", "zoobot", "zoobot_large"]
    bb_labels = [BACKBONE_LABELS[b] for b in backbones_order]

    # Build matrix (only 128-D for clean comparison)
    matrix = np.full((len(archs), len(backbones_order)), np.nan)
    annot = [[""]*len(backbones_order) for _ in range(len(archs))]

    for r in results:
        if r.get("embed_dim") != 128:
            continue
        arch = parse_arch(r["encoder"])
        bb = r.get("backbone", "simple")
        if arch in archs and bb in backbones_order:
            i = archs.index(arch)
            j = backbones_order.index(bb)
            lp = r["linear_probe_acc"]
            ifp = r.get("if_precision_at_k", 0)
            matrix[i, j] = lp
            annot[i][j] = f"LP: {lp:.3f}\nIF: {ifp:.2f}\n{format_params(r['n_params'])}"

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Custom colormap: red (bad) -> white -> blue (good)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("perf",
        ["#FCA5A5", "#FFFFFF", "#93C5FD"], N=256)

    # Mask NaN cells
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked, cmap=cmap, aspect="auto",
                   vmin=0.42, vmax=0.55)

    ax.set_xticks(range(len(bb_labels)))
    ax.set_xticklabels(bb_labels, fontweight="bold")
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs, fontweight="bold")

    # Annotate cells
    for i in range(len(archs)):
        for j in range(len(backbones_order)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center",
                        fontsize=7, color="black")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8, color="gray", style="italic")

    plt.colorbar(im, ax=ax, label="Linear Probe Accuracy", shrink=0.8)

    ax.set_title("Architecture × Backbone Interaction (128-D)\n"
                 "Pretrained backbones help AstroDistill but hurt RTF",
                 fontweight="bold")

    fig.tight_layout()
    save_fig(fig, output_dir, "fig_heatmap_interaction")


# ======================================================================
# FIGURE 8: Metric vs Params — connected line plot with error bars
# Mirrors Nabeel's Fig 2 (ROC AUC vs training set size) format.
# Shows each architecture as a line across its configurations.
# ======================================================================

def fig_metric_vs_params(results, output_dir):
    """
    Line plot: x = params (log scale), y = linear probe accuracy.
    One line per architecture, connecting its configurations.
    Error bars from linear_probe_std (cross-validation variance).
    Mirrors Nabeel's Figure 2 format.
    RULE: the shape of the line tells the story — RTF degrades
    with more params (backbones), while AstroDistill is flat.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (metric, title, std_key) in zip(axes, [
        ("linear_probe_acc", "Linear Probe Accuracy", "linear_probe_std"),
        ("if_precision_at_k", "IF Precision @ 20", None),
    ]):
        for arch, color in ARCH_COLORS.items():
            arch_results = [r for r in results if parse_arch(r["encoder"]) == arch]
            arch_results = sorted(arch_results, key=lambda r: r["n_params"])

            params = [r["n_params"] for r in arch_results]
            vals = [r.get(metric, 0) for r in arch_results]
            errs = [r.get(std_key, 0) if std_key else 0 for r in arch_results]

            # Plot line with error bars
            ax.errorbar(params, vals, yerr=errs if std_key else None,
                        fmt="o-", color=color, label=arch,
                        linewidth=1.5, markersize=5, capsize=3,
                        elinewidth=1, alpha=0.85)

            # Annotate each point with dim + backbone
            for r, x, y in zip(arch_results, params, vals):
                dim = r.get("embed_dim", "?")
                bb = r.get("backbone", "simple")
                label = f"{dim}D"
                if bb != "simple":
                    label += f"+{BACKBONE_LABELS.get(bb, bb)[:4]}"
                ax.annotate(label, (x, y), fontsize=5,
                            textcoords="offset points", xytext=(4, 4),
                            color=color, alpha=0.7)

        # Chance-level baseline
        draw_chance_line(ax, metric)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: format_params(int(x))))
        ax.set_xlabel("Parameters (log scale)", fontweight="bold")
        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, fontsize=8)

    fig.suptitle("Performance vs Model Size — Each Architecture Across All Configurations",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, output_dir, "fig_metric_vs_params")


# ======================================================================
# FIGURE 9: Concrete example — a real TDE that was flagged
# RULE: every paper also carries at least one real concrete
# example sitting next to the aggregate stats."
# ======================================================================

def fig_concrete_example(results, emb_dir, output_dir):
    """
    Shows a real object from the evaluation set.
    Left: 2D UMAP/PCA of RTF 128-D embeddings, highlighting the top-k
          anomalies from Isolation Forest, coloring TDEs distinctly.
    Right: table of the top-20 most anomalous objects with their
           ZTF names, classes, and anomaly scores.
    RULE: one real case grounds the aggregate statistics.
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest

    # Use the best model: RTF 128-D
    emb_path = os.path.join(emb_dir, "emb_rtf_128-d.npz")
    if not os.path.exists(emb_path):
        print(f"  SKIP: {emb_path} not found.")
        return

    data = np.load(emb_path, allow_pickle=True)
    emb = data["embeddings"]
    labels = data["labels_fine"]
    obj_ids = data["obj_ids"]

    # Run Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42)
    iso.fit(emb)
    scores = -iso.decision_function(emb)  # higher = more anomalous

    # Rank by anomaly score
    ranked_idx = np.argsort(scores)[::-1]
    top_k = 20

    # PCA for visualization (fast, deterministic)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb)

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

    # --- Left: scatter with anomalies highlighted ---
    ax_scatter = fig.add_subplot(gs[0])

    # Plot all points by class
    for cls, color in CLASS_COLORS.items():
        mask = labels == cls
        if mask.sum() == 0:
            continue
        ax_scatter.scatter(coords[mask, 0], coords[mask, 1],
                           c=color, s=10, alpha=0.4, edgecolors="none",
                           label=CLASS_SHORT.get(cls, cls))

    # Highlight top-k anomalies with rings
    top_coords = coords[ranked_idx[:top_k]]
    top_labels = labels[ranked_idx[:top_k]]
    [CLASS_COLORS.get(l, "gray") for l in top_labels]
    ax_scatter.scatter(top_coords[:, 0], top_coords[:, 1],
                       s=80, facecolors="none", edgecolors="black",
                       linewidths=1.5, zorder=5, label=f"Top-{top_k} anomalies")

    # Mark TDEs among top-k with a star
    tde_mask = top_labels == "Tidal Disruption Event"
    if tde_mask.sum() > 0:
        ax_scatter.scatter(top_coords[tde_mask, 0], top_coords[tde_mask, 1],
                           s=120, marker="*", c=CLASS_COLORS["Tidal Disruption Event"],
                           edgecolors="black", linewidths=0.5, zorder=6,
                           label="TDE (flagged)")

    ax_scatter.set_xlabel("PCA Component 1", fontweight="bold")
    ax_scatter.set_ylabel("PCA Component 2", fontweight="bold")
    ax_scatter.set_title("RTF 128-D Embedding Space\nTop-20 Isolation Forest Anomalies Circled",
                         fontweight="bold")
    ax_scatter.legend(fontsize=7, loc="lower left", frameon=True, ncol=2)

    # --- Right: table of top-20 anomalies ---
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    table_data = []
    for rank, idx in enumerate(ranked_idx[:top_k], 1):
        obj_id = str(obj_ids[idx])
        cls = CLASS_SHORT.get(str(labels[idx]), str(labels[idx]))
        score = scores[idx]
        is_tde = "YES" if str(labels[idx]) == "Tidal Disruption Event" else ""
        table_data.append([str(rank), obj_id, cls, f"{score:.3f}", is_tde])

    col_labels = ["Rank", "ZTF Object ID", "Class", "Anomaly\nScore", "TDE?"]
    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)

    # Color TDE rows
    for i, row in enumerate(table_data):
        if row[4] == "YES":
            for j in range(5):
                table[i+1, j].set_facecolor("#D1FAE5")  # light green

    # Header styling
    for j in range(5):
        table[0, j].set_facecolor("#1F2937")
        table[0, j].set_text_props(color="white", fontweight="bold")

    n_tde_found = sum(1 for idx in ranked_idx[:top_k]
                      if str(labels[idx]) == "Tidal Disruption Event")
    ax_table.set_title(f"Top-{top_k} Anomalies — {n_tde_found}/{top_k} are TDEs\n"
                       f"(IF Precision@{top_k} = {n_tde_found/top_k:.2f})",
                       fontweight="bold", fontsize=10)

    fig.suptitle("Concrete Example: What RTF Flags as Anomalous",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, output_dir, "fig_concrete_example")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the RTF paper."
    )
    parser.add_argument("--results", required=True,
                        help="Path to comparison_results.json")
    parser.add_argument("--emb-dir", required=True,
                        help="Directory containing emb_*.npz files")
    parser.add_argument("--output-dir", default="figures",
                        help="Output directory for figures")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP (requires umap-learn)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} model results from {args.results}")
    print(f"Output directory: {args.output_dir}")
    print()

    # --- Generate all figures ---
    print("[1/9] Full 16-model leaderboard...")
    fig_leaderboard(results, args.output_dir)

    print("\n[2/9] Dimensionality ablation (128/64/32)...")
    fig_dimensionality(results, args.output_dir)

    print("\n[3/9] Backbone impact on RTF...")
    fig_backbone_impact(results, args.output_dir)

    if not args.skip_umap:
        print("\n[4/9] UMAP projections (top 4 models)...")
        fig_umap(results, args.emb_dir, args.output_dir)
    else:
        print("\n[4/9] UMAP skipped (--skip-umap)")

    print("\n[5/9] Params vs Performance scatter...")
    fig_params_vs_performance(results, args.output_dir)

    print("\n[6/9] Radar chart (baseline comparison)...")
    fig_radar(results, args.output_dir)

    print("\n[7/9] Architecture × Backbone heatmap...")
    fig_heatmap(results, args.output_dir)

    print("\n[8/9] Metric vs Params line plot (Nabeel Fig 2 style)...")
    fig_metric_vs_params(results, args.output_dir)

    print("\n[9/9] Concrete example (real TDE anomaly case)...")
    fig_concrete_example(results, args.emb_dir, args.output_dir)

    print("\n" + "=" * 50)
    print("ALL FIGURES GENERATED.")
    print("=" * 50)


if __name__ == "__main__":
    main()
