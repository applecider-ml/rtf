"""
Pre-process raw alerts.npy files into compact NPZ files with photometry + metadata.

Converts data_ztf/{obj_id}/alerts.npy → output_dir/{split}/{obj_id}.npz
with keys:
    x:     (L, 37) float32 — [4 cont + 3 band + 28 meta] ready for model input
    label: int64

This is a one-time step that makes training fast (5s/epoch instead of 7min/epoch).

Usage:
    python src/preprocess_alerts.py \
        --alert-dir /fred/oz480/mcoughli/data_ztf \
        --splits /fred/oz480/mcoughli/AppleCider/photo_events/splits.csv \
        --labels-dir /fred/oz480/mcoughli/AppleCider/photo_events \
        --output-dir /fred/oz480/mcoughli/AppleCider/rtf/data \
        --horizon 100
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from dataset import ALERT_META_KEYS


def process_source(alerts_path, label, horizon=None):
    """Convert a single alerts.npy to the model input tensor."""
    alerts = np.load(alerts_path, allow_pickle=True)

    # Handle edge cases: 0-d array or empty
    if alerts.ndim == 0:
        return None
    if len(alerts) == 0:
        return None

    # Sort by JD
    jds = np.array([a["candidate"]["jd"] for a in alerts])
    order = np.argsort(jds)
    alerts = alerts[order]
    jds = jds[order]

    L = len(alerts)
    if L == 0:
        return None

    # Time features
    dt = (jds - jds[0]).astype(np.float32)
    dt_prev = np.zeros(L, dtype=np.float32)
    dt_prev[1:] = np.diff(jds).astype(np.float32)

    # Horizon cut
    if horizon is not None:
        mask = dt <= horizon
        alerts = alerts[mask]
        dt = dt[mask]
        dt_prev = dt_prev[mask]
        L = len(alerts)
        if L == 0:
            return None

    # Photometry
    fids = np.array([a["candidate"]["fid"] for a in alerts])
    magpsf = np.array([a["candidate"]["magpsf"] for a in alerts], dtype=np.float32)
    sigmapsf = np.array([a["candidate"]["sigmapsf"] for a in alerts], dtype=np.float32)

    logflux = (-0.4 * magpsf).astype(np.float32)
    logflux_err = (0.4 * sigmapsf).astype(np.float32)

    log_dt = np.log1p(dt)
    log_dt_prev = np.log1p(dt_prev)

    band_idx = (fids - 1).clip(0, 2).astype(np.int64)
    one_hot = np.eye(3, dtype=np.float32)[band_idx]

    base = np.column_stack([log_dt, log_dt_prev, logflux, logflux_err])

    # Alert metadata
    meta = np.zeros((L, len(ALERT_META_KEYS)), dtype=np.float32)
    for i, a in enumerate(alerts):
        cand = a["candidate"]
        for j, key in enumerate(ALERT_META_KEYS):
            val = cand.get(key, np.nan)
            if val is None or (isinstance(val, (int, float)) and val == -999):
                val = np.nan
            try:
                meta[i, j] = float(val)
            except (ValueError, TypeError):
                meta[i, j] = np.nan
    meta = np.nan_to_num(meta, nan=0.0)

    x = np.concatenate([base, one_hot, meta], axis=1).astype(np.float32)
    return x


def main():
    parser = argparse.ArgumentParser(description="Pre-process alerts.npy → compact NPZ")
    parser.add_argument("--alert-dir", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=float, default=100.0)
    args = parser.parse_args()

    # Read splits
    sources = {}
    with open(args.splits) as f:
        for row in csv.DictReader(f):
            sources[row["obj_id"]] = row["split"]

    labels_dir = Path(args.labels_dir)

    for split in ["train", "val", "test"]:
        out_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_dir, exist_ok=True)

    n_done, n_skip = 0, 0
    for oid, split in sorted(sources.items()):
        alerts_path = os.path.join(args.alert_dir, oid, "alerts.npy")
        if not os.path.exists(alerts_path):
            n_skip += 1
            continue

        # Get label
        label_file = labels_dir / split / f"{oid}.npz"
        if label_file.exists():
            label = int(np.load(label_file, allow_pickle=True)["label"])
        else:
            n_skip += 1
            continue

        try:
            x = process_source(alerts_path, label, horizon=args.horizon)
        except Exception as e:
            print(f"  WARNING: failed to process {oid}: {e}")
            n_skip += 1
            continue
        if x is None:
            n_skip += 1
            continue

        out_path = os.path.join(args.output_dir, split, f"{oid}.npz")
        np.savez_compressed(out_path, x=x, label=np.int64(label))
        n_done += 1

        if n_done % 1000 == 0:
            print(f"  Processed {n_done} sources ({n_skip} skipped)...")

    print(f"\nDone: {n_done} sources processed, {n_skip} skipped")
    for split in ["train", "val", "test"]:
        n = len(list(Path(os.path.join(args.output_dir, split)).glob("*.npz")))
        print(f"  {split}: {n} files")


if __name__ == "__main__":
    main()
