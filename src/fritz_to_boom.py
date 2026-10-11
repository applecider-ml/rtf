"""
Convert Fritz source NPZ files to the format boom-fit-batch expects.

Boom-fit-batch reads NPZ files with a 'data.npy' array of shape (N, 5+):
    Column 0: dt (days since first detection)
    Column 1: dt_prev (days since previous detection)
    Column 2: band_id (0=g, 1=r, 2=i)
    Column 3: logflux (log10 flux in µJy, AB ZP = 23.9)
    Column 4: logflux_err (log10 flux error)

Our Fritz NPZ files store a (L, 37) tensor where:
    Col 0: log1p(dt)
    Col 1: log1p(dt_prev)
    Col 2: logflux (already -0.4 * mag)
    Col 3: logflux_err (already 0.4 * magerr)
    Col 4-6: one-hot band (g, r, i)
    Col 7-36: 30 alert metadata keys

We need to invert log1p and reconstruct the 5-column format.

Usage:
    python fritz_to_boom.py \
        --fritz-dir ../fritz_sources/ \
        --output-dir ../boom_input/
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def convert_fritz_to_boom(fritz_npz_path, output_npz_path):
    """Convert a single Fritz NPZ to boom-fit-batch format.

    Returns number of valid detections, or 0 if failed.
    """
    data = np.load(fritz_npz_path, allow_pickle=True)
    x = data["x"]  # (L, 37)

    L = x.shape[0]
    if L < 5:
        return 0

    # Extract columns from the (L, 37) tensor
    log1p_dt = x[:, 0]  # log1p(dt) → dt = expm1(log1p_dt)
    log1p_dt_prev = x[:, 1]  # log1p(dt_prev) → dt_prev = expm1(log1p_dt_prev)
    logflux = x[:, 2]  # already log10-ish flux (= -0.4 * mag)
    logflux_err = x[:, 3]  # already flux error (= 0.4 * magerr)
    one_hot_band = x[:, 4:7]  # (L, 3) one-hot: g, r, i

    # Invert log1p to get actual dt values
    dt = np.expm1(log1p_dt).astype(np.float32)
    dt_prev = np.expm1(log1p_dt_prev).astype(np.float32)

    # Band ID from one-hot: argmax gives 0=g, 1=r, 2=i
    band_id = one_hot_band.argmax(axis=1).astype(np.float32)

    # Our logflux is -0.4 * mag. Boom expects log10(flux_µJy) where
    # mag = -2.5 * log10(flux) + 23.9, so log10(flux) = (23.9 - mag) / 2.5
    # We have -0.4 * mag, so mag = logflux / -0.4
    # log10(flux_µJy) = (23.9 - mag) / 2.5 = (23.9 + logflux/0.4) / 2.5
    #                 = 23.9/2.5 + logflux/(0.4*2.5) = 9.56 + logflux
    boom_logflux = (9.56 + logflux).astype(np.float32)

    # logflux_err: we have 0.4 * magerr. Boom expects in log10 flux space.
    # mag_err * 0.4 = our value, Boom expects mag_err / 2.5 = our_value / (0.4 * 2.5) = our_value
    # Actually: Boom's read_photometry_npz does: mag = -2.5 * logflux + 23.9, mag_err = 2.5 * logflux_err
    # So boom_logflux_err = our_magerr / 2.5 = (our_logflux_err / 0.4) / 2.5 = our_logflux_err / 1.0 = our_logflux_err
    # Wait, let's be precise. In fetch_fritz_sources.py:
    #   logflux = -0.4 * mag  (so mag = logflux / -0.4)
    #   logflux_err = 0.4 * magerr (so magerr = logflux_err / 0.4)
    # In boom main.rs read_photometry_npz:
    #   mag = -2.5 * data[3] + 23.9    (so data[3] = (23.9 - mag) / 2.5)
    #   mag_err = 2.5 * data[4]        (so data[4] = magerr / 2.5)
    # So: boom_logflux_err = magerr / 2.5 = (logflux_err / 0.4) / 2.5 = logflux_err
    boom_logflux_err = logflux_err.astype(np.float32)

    # Validate: filter out non-finite values
    valid = np.isfinite(boom_logflux) & np.isfinite(boom_logflux_err) & (boom_logflux_err > 0)
    if valid.sum() < 5:
        return 0

    # Stack into (N, 5) array
    boom_data = np.column_stack(
        [
            dt[valid],
            dt_prev[valid],
            band_id[valid],
            boom_logflux[valid],
            boom_logflux_err[valid],
        ]
    ).astype(np.float32)

    # Save as NPZ with key 'data'
    np.savez_compressed(output_npz_path, data=boom_data)
    return int(valid.sum())


def main():
    parser = argparse.ArgumentParser(description="Convert Fritz NPZ to boom-fit-batch format")
    parser.add_argument("--fritz-dir", required=True, help="Fritz sources directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for boom NPZ files")
    args = parser.parse_args()

    fritz_dir = Path(args.fritz_dir)
    output_dir = Path(args.output_dir)

    # Create fritz/ subdirectory (boom expects {input_dir}/{split}/{obj_id}.npz)
    split_dir = output_dir / "fritz"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Find all Fritz NPZ files
    npz_files = sorted(fritz_dir.glob("*.npz"))
    npz_files = [f for f in npz_files if f.stem != "fritz_manifest"]
    print("=== Fritz → Boom Converter ===")
    print(f"Input: {fritz_dir} ({len(npz_files)} files)")
    print(f"Output: {output_dir}")

    sources_rows = []
    n_success = 0
    n_skip = 0

    for idx, npz_path in enumerate(npz_files):
        obj_id = npz_path.stem
        out_path = split_dir / f"{obj_id}.npz"

        n_det = convert_fritz_to_boom(str(npz_path), str(out_path))
        if n_det > 0:
            sources_rows.append({"obj_id": obj_id, "split": "fritz"})
            n_success += 1
        else:
            n_skip += 1

        if (idx + 1) % 500 == 0:
            print(f"  [{idx + 1}/{len(npz_files)}] {n_success} converted, {n_skip} skipped")

    # Write sources.csv
    sources_csv = output_dir / "sources.csv"
    with open(sources_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["obj_id", "split"])
        writer.writeheader()
        writer.writerows(sources_rows)

    print("\n=== DONE ===")
    print(f"  Converted: {n_success}")
    print(f"  Skipped (<5 detections): {n_skip}")
    print(f"  Sources CSV: {sources_csv}")
    print(f"  NPZ dir: {split_dir}")


if __name__ == "__main__":
    main()
