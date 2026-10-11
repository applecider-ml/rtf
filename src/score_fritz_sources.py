"""
Score Fritz sources using the frozen RTF autoencoder + trained Isolation Forest.

Loads preprocessed Fritz source NPZ files, runs inference through the frozen model,
scores with the Isolation Forest, and outputs a ranked CSV of anomaly scores.

Usage:
    python score_fritz_sources.py \
        --model-path /path/to/best_model.pt \
        --if-path /path/to/isolation_forest.joblib \
        --fritz-dir /path/to/fritz_sources/ \
        --output anomaly_scores.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from model import LightCurveCompressor


def load_model(model_path, device):
    """Load the frozen RTF autoencoder."""
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
        }
    model = LightCurveCompressor(**model_kwargs)

    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v
    model.load_state_dict(clean_state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def infer_single(model, x_np, image_np, device):
    """Run inference on a single source.

    Args:
        model: frozen LightCurveCompressor
        x_np: (L, 37) numpy array
        image_np: (3, 63, 63) numpy array
        device: torch device

    Returns:
        embedding: (128,) numpy array
        recon_error: float
    """
    # Truncate to model max_len (257) if needed
    MAX_LEN = 257
    if x_np.shape[0] > MAX_LEN:
        x_np = x_np[:MAX_LEN]

    # Build batch of 1
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1, L, 37)
    pad_mask = torch.zeros(1, x.shape[1], dtype=torch.bool).to(device)  # no padding
    images = torch.from_numpy(image_np).unsqueeze(0).to(device)  # (1, 3, 63, 63)

    with torch.no_grad():
        emb = model.embed(x, pad_mask, images)  # (1, 128)
        recon_err = model.reconstruction_error(x, pad_mask, images)  # (1,)

    return emb[0].cpu().numpy(), recon_err[0].item()


def main():
    parser = argparse.ArgumentParser(description="Score Fritz sources with RTF + IF")
    parser.add_argument("--model-path", required=True, help="Path to best_model.pt")
    parser.add_argument("--if-path", required=True, help="Path to isolation_forest.joblib")
    parser.add_argument("--fritz-dir", required=True, help="Directory with Fritz source NPZ files")
    parser.add_argument("--output", default="anomaly_scores.csv", help="Output CSV path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Fritz Source Anomaly Scoring ===")
    print(f"Device: {device}")

    # ── Load model ──
    print("\n[1/4] Loading frozen model...")
    model = load_model(args.model_path, device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Load Isolation Forest ──
    print("\n[2/4] Loading Isolation Forest...")
    iso_forest = joblib.load(args.if_path)
    print(f"  IF loaded: {iso_forest.n_estimators} estimators")

    # ── Find Fritz source files ──
    print(f"\n[3/4] Scanning {args.fritz_dir}...")
    npz_files = sorted(Path(args.fritz_dir).glob("*.npz"))
    # Exclude the manifest file
    npz_files = [f for f in npz_files if f.stem != "fritz_manifest"]
    print(f"  Found {len(npz_files)} source files")

    if not npz_files:
        print("ERROR: No NPZ files found!")
        sys.exit(1)

    # ── Load manifest for group/classification info ──
    manifest = {}
    manifest_path = os.path.join(args.fritz_dir, "fritz_manifest.csv")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest[row["obj_id"]] = row

    # ── Run inference on each source ──
    print(f"\n[4/4] Running inference on {len(npz_files)} sources...")
    results = []
    t0 = time.time()

    for idx, npz_path in enumerate(npz_files):
        obj_id = npz_path.stem

        try:
            data = np.load(npz_path, allow_pickle=True)
            x = data["x"]  # (L, 37)
            images = data["images"]  # (3, 63, 63)

            if x.shape[0] < 3:
                print(f"  [{idx + 1}] {obj_id}: too few detections ({x.shape[0]}), skipping")
                continue

            # Get embedding + reconstruction error
            embedding, recon_error = infer_single(model, x, images, device)

            # Get Isolation Forest anomaly score
            # score_samples returns negative values; more negative = more anomalous
            if_score = iso_forest.score_samples(embedding.reshape(1, -1))[0]

            # Look up manifest info
            info = manifest.get(obj_id, {})
            group_name = info.get("group_name", str(data.get("group_name", "unknown")))
            classification = info.get("classification", "")
            n_det = x.shape[0]

            results.append(
                {
                    "obj_id": obj_id,
                    "group_name": group_name,
                    "n_detections": n_det,
                    "classification": classification,
                    "recon_error": round(recon_error, 6),
                    "if_score": round(if_score, 6),
                }
            )

        except Exception as e:
            print(f"  [{idx + 1}] {obj_id}: ERROR - {e}")
            continue

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx + 1}/{len(npz_files)}] {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(f"\n  Processed {len(results)} sources in {elapsed:.1f}s")

    if not results:
        print("ERROR: No results produced!")
        sys.exit(1)

    # ── Compute combined anomaly score ──
    # Normalize recon_error and IF score to [0, 1], then combine
    recon_errors = np.array([r["recon_error"] for r in results])
    if_scores = np.array([r["if_score"] for r in results])

    # Recon error: higher = more anomalous → normalize to [0, 1]
    recon_min, recon_max = recon_errors.min(), recon_errors.max()
    if recon_max > recon_min:
        recon_norm = (recon_errors - recon_min) / (recon_max - recon_min)
    else:
        recon_norm = np.zeros_like(recon_errors)

    # IF score: more negative = more anomalous → flip and normalize
    if_min, if_max = if_scores.min(), if_scores.max()
    if if_max > if_min:
        if_norm = (if_max - if_scores) / (if_max - if_min)
    else:
        if_norm = np.zeros_like(if_scores)

    # Combined score: 0.5 * recon + 0.5 * IF
    anomaly_scores = 0.5 * recon_norm + 0.5 * if_norm

    for i, r in enumerate(results):
        r["anomaly_score"] = round(float(anomaly_scores[i]), 6)

    # Sort by anomaly score descending (most anomalous first)
    results.sort(key=lambda r: r["anomaly_score"], reverse=True)

    # ── Write CSV ──
    fieldnames = [
        "obj_id",
        "group_name",
        "n_detections",
        "classification",
        "recon_error",
        "if_score",
        "anomaly_score",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved: {args.output}")

    # ── Print top-20 per group ──
    groups_seen = set()
    for r in results:
        groups_seen.add(r["group_name"])

    for group in sorted(groups_seen):
        group_results = [r for r in results if r["group_name"] == group]
        print(f"\n{'=' * 60}")
        print(f"  TOP 10 MOST ANOMALOUS — {group} ({len(group_results)} total)")
        print(f"{'=' * 60}")
        print(f"  {'Rank':<5} {'Object ID':<20} {'Score':<8} {'Recon':<10} {'IF':<10} {'Class'}")
        print(f"  {'-' * 5} {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 20}")
        for rank, r in enumerate(group_results[:10], 1):
            print(
                f"  {rank:<5} {r['obj_id']:<20} {r['anomaly_score']:<8.4f} "
                f"{r['recon_error']:<10.6f} {r['if_score']:<10.4f} "
                f"{r['classification'][:20]}"
            )

    print("\n=== DONE ===")
    print(f"  Total scored: {len(results)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
