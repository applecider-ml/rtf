"""
Pre-process raw alerts.npy files into compact NPZ files with photometry + metadata + images + GP features.

Converts data_ztf/{obj_id}/alerts.npy → output_dir/{split}/{obj_id}.npz
with keys:
    x:           (L, 37) float32 — [4 cont + 3 band + 28 meta] ready for model input
    images:      (L, 3, 63, 63) float32 — [science, template, difference] cutout stamps
    has_image:   (L,) float32
    gp_features: (N_GP,) float32 — GP-derived features from lightcurve-fitting (if --fit-gp)
    label:       int64

Usage:
    python src/preprocess_alerts.py \
        --alert-dir /fred/oz480/mcoughli/data_ztf \
        --splits /fred/oz480/mcoughli/AppleCider/photo_events/splits.csv \
        --labels-dir /fred/oz480/mcoughli/AppleCider/photo_events \
        --output-dir /fred/oz480/mcoughli/AppleCider/rtf/data \
        --horizon 100 \
        --fit-gp
"""

import argparse
import csv
import gzip
import io
import os
from pathlib import Path

import numpy as np

from dataset import ALERT_META_KEYS

STAMP_SIZE = 63

# Band name mapping: ZTF fid → lightcurve-fitting band name
FID_TO_BAND = {1: "ztfg", 2: "ztfr", 3: "ztfi"}


def compute_gp_features(alerts):
    """Run lightcurve-fitting's extract_features on alert photometry.

    Calls the Rust GP fitter via Python bindings — ~0.2ms per source.
    Returns a dict of feature_name → float, or None on failure.
    """
    try:
        from lightcurve_fitting import (
            build_flux_bands,
            build_mag_bands,
            extract_features,
        )
    except ImportError:
        return None

    jds = [float(a["candidate"]["jd"]) for a in alerts]
    mags = [float(a["candidate"]["magpsf"]) for a in alerts]
    errs = [float(a["candidate"]["sigmapsf"]) for a in alerts]
    bands = [FID_TO_BAND.get(a["candidate"]["fid"], "ztfr") for a in alerts]

    try:
        mag_bands = build_mag_bands(jds, mags, errs, bands)
        flux_bands = build_flux_bands(jds, mags, errs, bands)
        features = extract_features(mag_bands, flux_bands, "ztfr")
        return features
    except Exception:
        return None


def decode_stamp(stamp_bytes):
    """Decode a gzip-compressed FITS stamp to a numpy array."""
    from astropy.io import fits

    decompressed = gzip.decompress(stamp_bytes)
    with fits.open(io.BytesIO(decompressed), ignore_missing_simple=True) as hdu:
        return hdu[0].data.astype(np.float32)


def extract_cutouts(alert):
    """Extract 3-channel cutout (science, template, difference) from an alert.

    Returns (3, 63, 63) float32 array, or None if any stamp is missing/corrupt.
    """
    try:
        sci = decode_stamp(alert["cutoutScience"]["stampData"])
        tmpl = decode_stamp(alert["cutoutTemplate"]["stampData"])
        diff = decode_stamp(alert["cutoutDifference"]["stampData"])

        # Validate shapes
        if sci.shape != (STAMP_SIZE, STAMP_SIZE):
            return None
        if tmpl.shape != (STAMP_SIZE, STAMP_SIZE):
            return None
        if diff.shape != (STAMP_SIZE, STAMP_SIZE):
            return None

        # Replace NaN with 0
        sci = np.nan_to_num(sci, nan=0.0)
        tmpl = np.nan_to_num(tmpl, nan=0.0)
        diff = np.nan_to_num(diff, nan=0.0)

        return np.stack([sci, tmpl, diff], axis=0)  # (3, 63, 63)
    except Exception:
        return None


def process_source(alerts_path, label, horizon=None, fit_gp=False):
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

    # Cutout images: (L, 3, 63, 63)
    images = np.zeros((L, 3, STAMP_SIZE, STAMP_SIZE), dtype=np.float32)
    has_image = np.zeros(L, dtype=np.float32)
    for i, a in enumerate(alerts):
        cutout = extract_cutouts(a)
        if cutout is not None:
            images[i] = cutout
            has_image[i] = 1.0

    # GP features (optional): run lightcurve-fitting on the photometry
    gp_features = None
    if fit_gp:
        feat_dict = compute_gp_features(alerts)
        if feat_dict is not None:
            # Build canonical key list from a full-featured source (114 keys)
            # Use sorted keys from the LONGEST feature dict we've seen
            if not hasattr(process_source, "_canonical_keys"):
                process_source._canonical_keys = sorted(feat_dict.keys())
            elif len(feat_dict) > len(process_source._canonical_keys):
                process_source._canonical_keys = sorted(feat_dict.keys())
            canonical_keys = process_source._canonical_keys

            # Map features to canonical positions; missing → 0.0
            gp_vec = np.zeros(len(canonical_keys), dtype=np.float32)
            for j, key in enumerate(canonical_keys):
                val = feat_dict.get(key)
                if val is not None and np.isfinite(val):
                    gp_vec[j] = float(val)
            gp_features = (gp_vec, canonical_keys)

    return x, images, has_image, gp_features


def _process_one(task):
    """Worker function for parallel processing."""
    oid, split, alerts_path, label_file, output_dir, horizon, fit_gp = task
    try:
        label = int(np.load(label_file, allow_pickle=True)["label"])
    except Exception:
        return "skip"

    try:
        result = process_source(alerts_path, label, horizon=horizon, fit_gp=fit_gp)
    except Exception:
        return "skip"
    if result is None:
        return "skip"

    x, images, has_image, gp_features = result
    out_path = os.path.join(output_dir, split, f"{oid}.npz")
    save_dict = dict(x=x, images=images, has_image=has_image, label=np.int64(label))
    if gp_features is not None:
        gp_vec, gp_keys = gp_features
        save_dict["gp_features"] = gp_vec
        save_dict["gp_keys"] = np.array(gp_keys)
    np.savez_compressed(out_path, **save_dict)
    return "done"


def main():
    parser = argparse.ArgumentParser(description="Pre-process alerts.npy → compact NPZ")
    parser.add_argument("--alert-dir", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=float, default=100.0)
    parser.add_argument(
        "--fit-gp",
        action="store_true",
        help="Run lightcurve-fitting GP on each source (~0.2ms/source)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (0=serial)"
    )
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

    # Build task list, skipping already-processed and missing files
    tasks = []
    for oid, split in sorted(sources.items()):
        out_path = os.path.join(args.output_dir, split, f"{oid}.npz")
        if os.path.exists(out_path):
            continue
        alerts_path = os.path.join(args.alert_dir, oid, "alerts.npy")
        if not os.path.exists(alerts_path):
            continue
        label_file = str(labels_dir / split / f"{oid}.npz")
        if not os.path.exists(label_file):
            continue
        tasks.append(
            (
                oid,
                split,
                alerts_path,
                label_file,
                args.output_dir,
                args.horizon,
                args.fit_gp,
            )
        )

    print(f"Processing {len(tasks)} sources with {args.workers} workers...")

    if args.workers <= 1:
        # Serial
        n_done, n_skip = 0, 0
        for i, task in enumerate(tasks):
            status = _process_one(task)
            if status == "done":
                n_done += 1
            else:
                n_skip += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} ({n_done} done, {n_skip} skipped)")
    else:
        # Parallel
        from multiprocessing import Pool

        n_done, n_skip = 0, 0
        with Pool(args.workers) as pool:
            for i, status in enumerate(
                pool.imap_unordered(_process_one, tasks, chunksize=10)
            ):
                if status == "done":
                    n_done += 1
                else:
                    n_skip += 1
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(tasks)} ({n_done} done, {n_skip} skipped)")

    print(f"\nDone: {n_done} processed, {n_skip} skipped")
    for split in ["train", "val", "test"]:
        n = len(list(Path(os.path.join(args.output_dir, split)).glob("*.npz")))
        print(f"  {split}: {n} files")


if __name__ == "__main__":
    main()
