"""
Pack 1.74M individual NPZ files into ~87 large shard .pt files.

This eliminates the catastrophic GPFS file-open overhead that causes
1.74 million network metadata lookups per training epoch.

Each shard contains ~20,000 pre-processed samples with:
  - x_concat:  all sequences concatenated into one tensor (fast, no padding waste)
  - x_offsets: index array to slice individual sequences out
  - images:    stacked pre-extracted+normalized cutout stamps
  - labels:    integer label array
  - obj_ids:   list of object ID strings

Usage:
    python pack_shards.py \
        --input-dirs /path/to/day1,/path/to/day2,... \
        --output-dir /path/to/shards \
        --shard-size 20000 \
        --workers 16
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

STAMP_SIZE = 63


def process_single_npz(npz_path):
    """Load one NPZ and extract raw data for sharding.

    Image extraction + normalization is IDENTICAL to MetaNPZDataset._load_item
    so that training produces bit-identical results.

    Returns (x, image, label, seq_len, obj_id) or None on failure.
    """
    try:
        npz = np.load(npz_path, allow_pickle=True)
        x = npz["x"].astype(np.float32)
        label = int(npz["label"])
        L = x.shape[0]

        if L == 0:
            return None

        # Extract image: identical to MetaNPZDataset._load_item
        image = np.zeros((3, STAMP_SIZE, STAMP_SIZE), dtype=np.float32)
        if "images" in npz and "has_image" in npz:
            all_images = npz["images"][:L].astype(np.float32)
            all_has_image = npz["has_image"][:L].astype(np.float32)
            valid_idx = np.where(all_has_image > 0)[0]
            if len(valid_idx) > 0:
                img = all_images[valid_idx[-1]]  # most recent valid stamp (3, 63, 63)
                # Normalize per-channel: identical to MetaNPZDataset._load_item
                for c in range(3):
                    ch = img[c].astype(np.float64)
                    p1, p99 = np.percentile(ch, [1, 99])
                    ch = np.clip(ch, p1, p99)
                    std = ch.std()
                    if std > 1e-6:
                        image[c] = ((ch - ch.mean()) / std).astype(np.float32)
                    else:
                        image[c] = 0.0

        obj_id = Path(npz_path).stem
        return (x, image, label, L, obj_id)

    except Exception:
        return None


def save_shard(buffer, shard_idx, output_dir):
    """Pack a buffer of processed samples into a single .pt shard file."""
    # Concatenate all variable-length x tensors + build offset index
    x_list = [torch.from_numpy(r[0]) for r in buffer]
    lengths = [t.shape[0] for t in x_list]
    x_concat = torch.cat(x_list, dim=0)
    x_offsets = torch.tensor([0] + lengths).cumsum(0)

    # Stack images into one contiguous tensor
    images = torch.stack([torch.from_numpy(r[1]) for r in buffer])
    labels = torch.tensor([r[2] for r in buffer], dtype=torch.long)
    obj_ids = [r[4] for r in buffer]

    shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
    torch.save(
        {
            "x_concat": x_concat,
            "x_offsets": x_offsets,
            "images": images,
            "labels": labels,
            "obj_ids": obj_ids,
        },
        shard_path,
    )

    size_mb = os.path.getsize(shard_path) / (1024 * 1024)
    print(
        f"  [SHARD {shard_idx:3d}] {len(buffer):,} samples, "
        f"x_concat={tuple(x_concat.shape)}, "
        f"images={tuple(images.shape)}, "
        f"{size_mb:.0f} MB → {os.path.basename(shard_path)}",
        flush=True,
    )
    return shard_path


def main():
    parser = argparse.ArgumentParser(description="Pack NPZ files into .pt shards")
    parser.add_argument(
        "--input-dirs", required=True, help="Comma-separated list of data directories"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write shard .pt files")
    parser.add_argument(
        "--shard-size", type=int, default=20000, help="Number of samples per shard (default: 20000)"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of parallel workers (default: 16)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    shard_size = args.shard_size
    workers = args.workers

    t_start = time.time()

    # ── Step 1: Collect all NPZ file paths ──
    print("=" * 60)
    print("Step 1: Scanning for NPZ files...")
    print("=" * 60, flush=True)

    all_files = []
    for d in args.input_dirs.split(","):
        d = d.strip()
        # Check for infer/ subdirectory (standard layout from preprocess_alerts.py)
        infer_dir = os.path.join(d, "infer")
        target = infer_dir if os.path.isdir(infer_dir) else d
        npz_files = sorted(str(f) for f in Path(target).glob("*.npz"))
        print(f"  [SCAN] {target}: {len(npz_files):,} NPZ files", flush=True)
        all_files.extend(npz_files)

    total = len(all_files)
    n_shards_expected = (total + shard_size - 1) // shard_size
    print(f"\n  Total: {total:,} NPZ files")
    print(f"  Shard size: {shard_size:,}")
    print(f"  Expected shards: ~{n_shards_expected}")
    print(f"  Workers: {workers}")

    # ── Step 2: Process files and pack into shards ──
    print(f"\n{'=' * 60}")
    print(f"Step 2: Processing + packing with {workers} workers...")
    print(f"{'=' * 60}", flush=True)

    buffer = []
    shard_idx = 0
    processed = 0
    valid = 0
    failed = 0
    shard_paths = []

    with Pool(workers) as pool:
        for result in pool.imap(process_single_npz, all_files, chunksize=200):
            processed += 1

            if result is not None:
                buffer.append(result)
                valid += 1
            else:
                failed += 1

            # Progress report every 5000 files
            if processed % 5000 == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed
                eta = (total - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed:>8,}/{total:,}] "
                    f"{valid:,} valid, {failed:,} failed | "
                    f"{rate:.0f} files/sec | "
                    f"ETA: {eta / 60:.1f} min | "
                    f"buffer: {len(buffer):,}",
                    flush=True,
                )

            # Flush buffer to shard when full
            if len(buffer) >= shard_size:
                shard_paths.append(save_shard(buffer[:shard_size], shard_idx, output_dir))
                buffer = buffer[shard_size:]
                shard_idx += 1

    # Save remaining samples as final shard
    if buffer:
        shard_paths.append(save_shard(buffer, shard_idx, output_dir))
        shard_idx += 1

    # ── Step 3: Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print("SHARDING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Total files processed: {processed:,}")
    print(f"  Valid samples packed:  {valid:,}")
    print(f"  Failed/skipped:       {failed:,}")
    print(f"  Shards created:       {shard_idx}")
    print(f"  Output directory:     {output_dir}")
    print(f"  Total time:           {elapsed / 60:.1f} minutes")
    print(f"  Average rate:         {processed / elapsed:.0f} files/sec")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
