#!/usr/bin/env python3
"""Validate TEMPO ONNX model against the labeled test set.

Produces:
  - Classification report (accuracy, precision, recall, F1 per class)
  - Confusion matrix (PNG)
  - Uncertainty analysis: distribution for correct vs incorrect predictions
  - Per-sample CSV with predictions, probabilities, and uncertainty

Usage (on Delta):
    python validate_tempo.py \
        --onnx /work/hdd/bcrv/kmajithia/tempo/onnx/tempo_classifier.onnx \
        --data-dir /projects/bcrv/ffontinelenunes/transformer/tempo/data/photo_events \
        --output-dir /work/hdd/bcrv/kmajithia/tempo/validation
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd

# ---------------------------------------------------------------------------
# Add TEMPO codebase for preprocessing imports
# ---------------------------------------------------------------------------
TEMPO_ROOT = os.environ.get(
    "TEMPO_ROOT",
    "/projects/bcrv/ffontinelenunes/transformer/tempo",
)
if TEMPO_ROOT not in sys.path:
    sys.path.insert(0, TEMPO_ROOT)

from photometry_edl.data import (
    FeatureStats,
    _global_features_from_sequence,
    build_event_tensor,
    preprocess_photometry_array,
    read_manifest_csv,
)
from photometry_edl.taxonomy import get_taxonomy

# ---------------------------------------------------------------------------
# Config matching the best checkpoint
# ---------------------------------------------------------------------------
MAX_LEN = 257
BAND_MODE = "embed"
HORIZON_DAYS = 100.0
DROP_I_BAND = True
GLOBAL_FEATURE_SET = "physics"
TAXONOMY_PRESET = "transient_variable_5c"
NUM_CLASSES = 5


def load_and_preprocess(filepath, stats):
    """Load a single .npy/.npz, preprocess, return (x_norm, global_feat, pad_mask)."""
    raw = np.load(filepath, allow_pickle=False)
    arr = raw["data"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
    arr = np.asarray(arr, dtype=np.float32)

    # Preprocess: horizon cut + drop i-band
    arr = preprocess_photometry_array(
        arr, horizon_days=HORIZON_DAYS, drop_i_band=DROP_I_BAND
    )

    # Build event tensor (unnormalized)
    import torch
    x_raw = build_event_tensor(arr, band_mode=BAND_MODE)

    # Compute global features from unnormalized sequence
    g = _global_features_from_sequence(
        x_raw, band_mode=BAND_MODE, feature_set=GLOBAL_FEATURE_SET
    )

    # Truncate to MAX_LEN
    seq_len = x_raw.shape[0]
    if seq_len > MAX_LEN:
        x_raw = x_raw[-MAX_LEN:]
        seq_len = MAX_LEN

    # Normalize continuous channels
    mean = stats.mean.view(1, 4)
    std = stats.std.view(1, 4)
    x_norm = x_raw.clone()
    x_norm[:, :4] = (x_raw[:, :4] - mean) / (std + 1e-8)

    # Pad to MAX_LEN
    if seq_len < MAX_LEN:
        pad_len = MAX_LEN - seq_len
        pad_tensor = torch.zeros(pad_len, x_norm.shape[1])
        x_norm = torch.cat([x_norm, pad_tensor], dim=0)

    # Build pad mask
    pad_mask = torch.zeros(MAX_LEN, dtype=torch.bool)
    pad_mask[seq_len:] = True

    # Add batch dim: (1, 257, 5), (1, 257), (1, 24)
    x_np = x_norm.unsqueeze(0).numpy()
    mask_np = pad_mask.unsqueeze(0).numpy()
    g_np = g.unsqueeze(0).numpy()

    return x_np, mask_np, g_np, seq_len


def evidence_to_probs(evidence):
    """Convert evidence to Dirichlet probabilities + uncertainty."""
    alpha = evidence + 1.0
    S = alpha.sum()
    probs = alpha / S
    uncertainty = NUM_CLASSES / S
    return probs, uncertainty, alpha, S


def main():
    parser = argparse.ArgumentParser(description="Validate TEMPO ONNX on test set")
    parser.add_argument(
        "--onnx",
        default="/work/hdd/bcrv/kmajithia/tempo/onnx/tempo_classifier.onnx",
    )
    parser.add_argument(
        "--data-dir",
        default="/projects/bcrv/ffontinelenunes/transformer/tempo/data/photo_events",
    )
    parser.add_argument(
        "--output-dir",
        default="/work/hdd/bcrv/kmajithia/tempo/validation",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # ------------------------------------------------------------------
    # Load taxonomy
    # ------------------------------------------------------------------
    tax = get_taxonomy(TAXONOMY_PRESET)
    class_names = tax.broad_classes  # ["SNI", "SNII", "TDE", "AGN", "CV"]
    id2broad = tax.id2broad_id  # subclass_id -> broad_id
    print(f"Taxonomy: {TAXONOMY_PRESET}")
    print(f"Classes: {class_names}")

    # ------------------------------------------------------------------
    # Load feature stats
    # ------------------------------------------------------------------
    stats = FeatureStats.load(data_dir / "feature_stats_day100.npz")
    print(f"Feature stats mean: {stats.mean.numpy()}")
    print(f"Feature stats std:  {stats.std.numpy()}")

    # ------------------------------------------------------------------
    # Load test manifest
    # ------------------------------------------------------------------
    manifest = read_manifest_csv(data_dir / "manifest_test.csv", data_dir=data_dir)
    print(f"\nTest samples: {len(manifest)}")

    # ------------------------------------------------------------------
    # Load ONNX model
    # ------------------------------------------------------------------
    print(f"\nLoading ONNX: {args.onnx}")
    sess = ort.InferenceSession(args.onnx)
    input_names = [inp.name for inp in sess.get_inputs()]
    print(f"ONNX inputs: {input_names}")

    # ------------------------------------------------------------------
    # Run inference on all test samples
    # ------------------------------------------------------------------
    results = []
    n_errors = 0

    for idx in range(len(manifest)):
        row = manifest.iloc[idx]
        filepath = row.filepath
        y_sub = int(row.label)
        y_broad = int(id2broad[y_sub])

        try:
            x_np, mask_np, g_np, seq_len = load_and_preprocess(filepath, stats)
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  [WARN] Failed to load {filepath}: {e}")
            continue

        # Run ONNX
        feed = {
            input_names[0]: x_np,
            input_names[1]: mask_np,
            input_names[2]: g_np,
        }
        evidence = sess.run(None, feed)[0][0]  # (5,)

        probs, uncertainty, _alpha, S = evidence_to_probs(evidence)
        pred_class = int(np.argmax(probs))

        results.append({
            "filepath": str(filepath),
            "true_subclass": y_sub,
            "true_broad": y_broad,
            "true_label": class_names[y_broad],
            "pred_class": pred_class,
            "pred_label": class_names[pred_class],
            "correct": int(pred_class == y_broad),
            "seq_len": seq_len,
            "uncertainty": float(uncertainty),
            "dirichlet_strength": float(S),
            "max_prob": float(probs.max()),
            **{f"prob_{class_names[i]}": float(probs[i]) for i in range(NUM_CLASSES)},
            **{f"evidence_{class_names[i]}": float(evidence[i]) for i in range(NUM_CLASSES)},
        })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(manifest)}")

    print(f"\nProcessed: {len(results)}, Errors: {n_errors}")

    # ------------------------------------------------------------------
    # Save per-sample CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = out_dir / "tempo_validation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------
    y_true = df["true_broad"].values
    y_pred = df["pred_class"].values

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    accuracy = accuracy_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3
    )

    print(f"\n{'='*60}")
    print("TEMPO ONNX Validation Results")
    print(f"{'='*60}")
    print(f"Test samples:       {len(df)}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced accuracy:  {bal_accuracy:.4f}")
    print(f"\n{report}")

    # Save report
    report_path = out_dir / "tempo_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("TEMPO ONNX Validation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"ONNX model: {args.onnx}\n")
        f.write(f"Test samples: {len(df)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced accuracy: {bal_accuracy:.4f}\n\n")
        f.write(f"{report}\n")
    print(f"Saved: {report_path}")

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("TEMPO ONNX — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    fig.colorbar(im, ax=ax)

    # Annotate cells
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    fig.tight_layout()
    cm_path = out_dir / "tempo_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {cm_path}")

    # ------------------------------------------------------------------
    # Uncertainty analysis
    # ------------------------------------------------------------------
    correct = df[df["correct"] == 1]
    incorrect = df[df["correct"] == 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TEMPO — Uncertainty Analysis", fontsize=16, fontweight="bold")

    # 1. Uncertainty distribution: correct vs incorrect
    ax = axes[0, 0]
    ax.hist(correct["uncertainty"], bins=50, alpha=0.7, label=f"Correct (n={len(correct)})", color="#2196F3")
    ax.hist(incorrect["uncertainty"], bins=50, alpha=0.7, label=f"Incorrect (n={len(incorrect)})", color="#F44336")
    ax.set_xlabel("Uncertainty (K/S)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty: Correct vs Incorrect")
    ax.legend()

    # 2. Max probability distribution
    ax = axes[0, 1]
    ax.hist(correct["max_prob"], bins=50, alpha=0.7, label="Correct", color="#2196F3")
    ax.hist(incorrect["max_prob"], bins=50, alpha=0.7, label="Incorrect", color="#F44336")
    ax.set_xlabel("Max Class Probability")
    ax.set_ylabel("Count")
    ax.set_title("Confidence: Correct vs Incorrect")
    ax.legend()

    # 3. Per-class uncertainty (box plot)
    ax = axes[1, 0]
    class_uncertainties = [df[df["true_broad"] == i]["uncertainty"].values for i in range(NUM_CLASSES)]
    bp = ax.boxplot(class_uncertainties, labels=class_names, patch_artist=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("True Class")
    ax.set_ylabel("Uncertainty")
    ax.set_title("Uncertainty by True Class")

    # 4. Accuracy vs uncertainty threshold
    ax = axes[1, 1]
    thresholds = np.linspace(0, df["uncertainty"].quantile(0.99), 50)
    accs = []
    coverages = []
    for t in thresholds:
        mask = df["uncertainty"] <= t
        if mask.sum() == 0:
            accs.append(0)
        else:
            accs.append(df.loc[mask, "correct"].mean())
        coverages.append(mask.mean())
    ax.plot(coverages, accs, "b-", linewidth=2)
    ax.set_xlabel("Coverage (fraction of samples retained)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy–Coverage Trade-off")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    ua_path = out_dir / "tempo_uncertainty_analysis.png"
    fig.savefig(ua_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {ua_path}")

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    summary = {
        "onnx_model": args.onnx,
        "test_samples": len(df),
        "errors": n_errors,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(bal_accuracy, 4),
        "mean_uncertainty_correct": round(float(correct["uncertainty"].mean()), 4),
        "mean_uncertainty_incorrect": round(float(incorrect["uncertainty"].mean()), 4) if len(incorrect) > 0 else None,
        "per_class": {},
    }
    for i, name in enumerate(class_names):
        cls_mask = df["true_broad"] == i
        cls_df = df[cls_mask]
        if len(cls_df) > 0:
            summary["per_class"][name] = {
                "support": len(cls_df),
                "accuracy": round(float((cls_df["pred_class"] == i).mean()), 4),
                "mean_uncertainty": round(float(cls_df["uncertainty"].mean()), 4),
                "median_uncertainty": round(float(cls_df["uncertainty"].median()), 4),
            }

    summary_path = out_dir / "tempo_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # ------------------------------------------------------------------
    # Print high-uncertainty examples
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Top 10 Most Uncertain Predictions")
    print(f"{'='*60}")
    top_uncertain = df.nlargest(10, "uncertainty")
    for _, row in top_uncertain.iterrows():
        marker = "✗" if not row["correct"] else "✓"
        print(
            f"  {marker} True: {row['true_label']:6s} → Pred: {row['pred_label']:6s} "
            f"| uncertainty={row['uncertainty']:.4f} "
            f"| max_prob={row['max_prob']:.4f} "
            f"| S={row['dirichlet_strength']:.1f}"
        )

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
