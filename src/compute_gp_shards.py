"""
Add boom-astro GP features to existing training shards.

Reads each shard .pt file, reconstructs raw photometry (times, mags, errors, bands)
from the x_concat tensor, calls lightcurve_fitting.extract_features() to compute
~80 GP + parametric + thermal features per source, and saves new shards with
gp_features included.

Input shards:  {input_dir}/shard_*.pt   (from pack_shards.py)
Output shards: {output_dir}/shard_*.pt  (same format + gp_features tensor)

Usage:
    python src/compute_gp_shards.py \
        --input-dir /work/hdd/bcrv/kmajithia/uw_tensors/shards \
        --output-dir /work/hdd/bcrv/kmajithia/uw_tensors/shards_gp \
        --workers 4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Band index → ZTF band name (matches preprocess_alerts.py)
IDX_TO_BAND = {0: "ztfg", 1: "ztfr", 2: "ztfi"}


def reconstruct_photometry(x_seq):
    """Reconstruct raw photometry arrays from a single source's x tensor.

    The x tensor has columns (from preprocess_alerts.py):
        0: log1p(dt)          - log1p of elapsed days since first obs
        1: log1p(dt_prev)     - log1p of time since previous obs
        2: logflux            - = -0.4 * magpsf
        3: logflux_err        - = 0.4 * sigmapsf
        4: band_g (one-hot)
        5: band_r (one-hot)
        6: band_i (one-hot)
        7+: metadata (ignored)

    Returns:
        times:  list[float] - JD-like times (days)
        mags:   list[float] - magnitudes
        errs:   list[float] - magnitude errors
        bands:  list[str]   - band names ("ztfg", "ztfr", "ztfi")
    """
    L = x_seq.shape[0]
    if L < 3:
        return None

    # Reconstruct absolute times from log1p(dt)
    # dt = jd - jd[0], stored as log1p(dt)
    log_dt = x_seq[:, 0].numpy().astype(np.float64)
    dt = np.expm1(log_dt)  # inverse of log1p → days since first obs

    # boom-astro expects JD-like times — add an arbitrary reference JD
    # (the absolute value doesn't matter, only relative spacing)
    t0_jd = 2460000.0
    times = (t0_jd + dt).tolist()

    # Reconstruct magnitudes from logflux = -0.4 * magpsf
    logflux = x_seq[:, 2].numpy().astype(np.float64)
    logflux_err = x_seq[:, 3].numpy().astype(np.float64)

    mags = (logflux / -0.4).tolist()  # mag = logflux / -0.4
    errs = (logflux_err / 0.4).tolist()  # sigmapsf = logflux_err / 0.4

    # Reconstruct bands from one-hot encoding
    band_onehot = x_seq[:, 4:7].numpy()
    band_indices = band_onehot.argmax(axis=1)
    bands = [IDX_TO_BAND.get(int(bi), "ztfr") for bi in band_indices]

    # Filter out any non-finite or unreasonable values
    valid = []
    for i in range(L):
        if np.isfinite(mags[i]) and np.isfinite(errs[i]) and errs[i] > 0 and 10.0 < mags[i] < 25.0:
            valid.append(i)

    if len(valid) < 3:
        return None

    times = [times[i] for i in valid]
    mags = [mags[i] for i in valid]
    errs = [errs[i] for i in valid]
    bands = [bands[i] for i in valid]

    return times, mags, errs, bands


def compute_features_for_shard(shard_path, output_dir):
    """Process one shard: extract GP features for all sources, save new shard."""
    from lightcurve_fitting import (
        build_flux_bands,
        build_mag_bands,
        extract_features,
    )

    shard = torch.load(shard_path, weights_only=False, map_location="cpu")
    n = len(shard["labels"])
    shard_name = os.path.basename(shard_path)

    print(f"  Processing {shard_name}: {n} sources...", flush=True)
    t0 = time.time()

    # First pass: compute features for all sources and collect canonical keys
    all_feature_dicts = []
    all_keys = set()
    n_phot_fail = 0
    n_extract_fail = 0
    errors_printed = 0

    for i in range(n):
        start = shard["x_offsets"][i].item()
        end = shard["x_offsets"][i + 1].item()
        x_seq = shard["x_concat"][start:end]

        phot = reconstruct_photometry(x_seq)
        if phot is None:
            all_feature_dicts.append(None)
            n_phot_fail += 1
            continue

        times, mags, errs, bands = phot

        try:
            mag_bands = build_mag_bands(times, mags, errs, bands)
            flux_bands = build_flux_bands(times, mags, errs, bands)
            features = extract_features(mag_bands, flux_bands, "ztfr")
            all_feature_dicts.append(features)
            all_keys.update(features.keys())
        except Exception as e:
            all_feature_dicts.append(None)
            n_extract_fail += 1
            if errors_printed < 3:
                print(f"    [ERROR] Source {i}: {type(e).__name__}: {e}", flush=True)
                print(
                    f"      L={len(times)}, mags[0:3]={mags[:3]}, "
                    f"errs[0:3]={errs[:3]}, bands[0:3]={bands[:3]}",
                    flush=True,
                )
                errors_printed += 1

    # Print diagnostics for first shard
    print(
        f"    {shard_name}: phot_fail={n_phot_fail}, extract_fail={n_extract_fail}, "
        f"success={n - n_phot_fail - n_extract_fail}",
        flush=True,
    )

    # Build canonical key order (sorted for reproducibility)
    canonical_keys = sorted(all_keys)
    n_gp = len(canonical_keys)
    print(f"    {shard_name}: {n_gp} GP feature keys identified", flush=True)

    # Second pass: build feature matrix
    gp_matrix = np.zeros((n, n_gp), dtype=np.float32)
    n_success = 0

    for i, feat_dict in enumerate(all_feature_dicts):
        if feat_dict is None:
            continue
        n_success += 1
        for j, key in enumerate(canonical_keys):
            val = feat_dict.get(key)
            if val is not None and np.isfinite(val):
                gp_matrix[i, j] = float(val)

    # Clip extreme values and handle NaN/inf
    gp_matrix = np.nan_to_num(gp_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    gp_matrix = np.clip(gp_matrix, -100.0, 100.0)

    elapsed = time.time() - t0

    # Save new shard with GP features added
    out_path = os.path.join(output_dir, shard_name)
    new_shard = {
        "x_concat": shard["x_concat"],
        "x_offsets": shard["x_offsets"],
        "images": shard["images"],
        "labels": shard["labels"],
        "obj_ids": shard["obj_ids"],
        "gp_features": torch.from_numpy(gp_matrix),
        "gp_keys": canonical_keys,
    }
    torch.save(new_shard, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(
        f"    {shard_name}: {n_success}/{n} sources fitted, "
        f"{n_gp} features, {elapsed:.1f}s, {size_mb:.0f} MB",
        flush=True,
    )

    return n, n_success, n_gp, canonical_keys


def main():
    parser = argparse.ArgumentParser(description="Add GP features to existing training shards")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing shard_*.pt files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write new shards with GP features",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (each worker processes one shard)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    shard_files = sorted(Path(args.input_dir).glob("shard_*.pt"))
    if not shard_files:
        print(f"ERROR: No shard files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(shard_files)} shards in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print("=" * 60, flush=True)

    t_start = time.time()

    if args.workers <= 1:
        # Serial processing
        total_sources = 0
        total_fitted = 0
        all_canonical_keys = None
        for sf in shard_files:
            n, n_success, n_gp, keys = compute_features_for_shard(str(sf), args.output_dir)
            total_sources += n
            total_fitted += n_success
            if all_canonical_keys is None:
                all_canonical_keys = set(keys)
            else:
                all_canonical_keys.update(keys)
    else:
        # Parallel: each worker handles one shard
        from multiprocessing import Pool

        tasks = [(str(sf), args.output_dir) for sf in shard_files]
        total_sources = 0
        total_fitted = 0
        all_canonical_keys = set()

        with Pool(args.workers) as pool:
            results = pool.starmap(compute_features_for_shard, tasks)
            for n, n_success, n_gp, keys in results:
                total_sources += n
                total_fitted += n_success
                all_canonical_keys.update(keys)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print("GP FEATURE COMPUTATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Shards processed: {len(shard_files)}")
    print(f"  Total sources:    {total_sources:,}")
    print(
        f"  Successfully fit: {total_fitted:,} ({100 * total_fitted / max(1, total_sources):.1f}%)"
    )
    print(f"  GP feature dim:   {len(all_canonical_keys) if all_canonical_keys else 0}")
    print(f"  Total time:       {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Output:           {args.output_dir}")
    print(f"{'=' * 60}", flush=True)

    # Save canonical key mapping for later use
    keys_path = os.path.join(args.output_dir, "gp_keys.txt")
    if all_canonical_keys:
        with open(keys_path, "w") as f:
            f.writelines(f"{key}\n" for key in sorted(all_canonical_keys))
        print(f"  Key mapping saved to {keys_path}")


if __name__ == "__main__":
    main()
