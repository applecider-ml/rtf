"""
Extract 128-dim reference embeddings from all training shards + train Isolation Forest.

This script streams through all 88 shards, passes each batch through the frozen
autoencoder to get embeddings and reconstruction errors, then trains an
IsolationForest on the full embedding space to define "normal."

Usage:
    python extract_reference.py \
        --model-path /path/to/best_model.pt \
        --shards-dir /tmp/shards \
        --output-dir /path/to/output
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ShardedDataset, collate_fn
from model import LightCurveCompressor


def main():
    parser = argparse.ArgumentParser(description="Extract reference embeddings + train IF")
    parser.add_argument("--model-path", required=True, help="Path to best_model.pt")
    parser.add_argument("--shards-dir", required=True, help="Directory with shard_*.pt files")
    parser.add_argument("--output-dir", required=True, help="Where to save outputs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--if-estimators", type=int, default=200, help="IsolationForest n_estimators"
    )
    parser.add_argument("--if-contamination", type=float, default=0.05, help="IF contamination")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Reference Embedding Extraction ===")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Shards: {args.shards_dir}")

    # ── Load model ──
    print("\n[1/4] Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and wrapped checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_kwargs = checkpoint.get("model_kwargs", {})
    else:
        state_dict = checkpoint
        model_kwargs = {}

    # Build model with same architecture as training
    if not model_kwargs:
        model_kwargs = {
            "mode": "ae",
            "latent_dim": 128,
            "in_channels": 37,
            "d_model": 256,
        }
    model = LightCurveCompressor(**model_kwargs)

    # Strip 'module.' prefix if saved from DataParallel
    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v
    model.load_state_dict(clean_state, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Create streaming dataset ──
    print("\n[2/4] Preparing streaming dataset...")
    dataset = ShardedDataset(
        shards_dir=args.shards_dir,
        use_images=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    total_samples = len(dataset)
    print(f"  ~{total_samples:,} samples to process")

    # ── Extract embeddings ──
    print("\n[3/4] Extracting embeddings (this takes ~30-40 min)...")
    all_embeddings = []
    all_recon_errors = []
    all_obj_ids = []
    processed = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            images = batch.get("images")
            if images is not None:
                images = images.to(device)

            # Get 128-dim embedding
            emb = model.embed(x, pad_mask, images)
            # Get per-sample reconstruction error
            recon_err = model.reconstruction_error(x, pad_mask, images)

            all_embeddings.append(emb.cpu().numpy())
            all_recon_errors.append(recon_err.cpu().numpy())
            all_obj_ids.extend(batch["obj_ids"])

            processed += len(x)
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (total_samples - processed) / rate if rate > 0 else 0
                print(
                    f"    [{processed:,}/{total_samples:,}] "
                    f"{rate:.0f} samples/sec | ETA: {eta / 60:.1f} min"
                )

    embeddings = np.concatenate(all_embeddings, axis=0)
    recon_errors = np.concatenate(all_recon_errors, axis=0)
    obj_ids = np.array(all_obj_ids)

    elapsed = time.time() - t0
    print(f"  Done! {len(embeddings):,} embeddings in {elapsed / 60:.1f} min")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Recon error: mean={recon_errors.mean():.6f}, std={recon_errors.std():.6f}")

    # Save embeddings
    emb_path = os.path.join(args.output_dir, "reference_embeddings.npz")
    np.savez_compressed(
        emb_path,
        embeddings=embeddings,
        recon_errors=recon_errors,
        obj_ids=obj_ids,
    )
    print(f"  Saved: {emb_path} ({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    # ── Train Isolation Forest ──
    print(f"\n[4/4] Training Isolation Forest (n_estimators={args.if_estimators})...")
    import joblib
    from sklearn.ensemble import IsolationForest

    t1 = time.time()
    iso_forest = IsolationForest(
        n_estimators=args.if_estimators,
        contamination=args.if_contamination,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    # Subsample if dataset is huge (>500K) for IF training speed
    if len(embeddings) > 500_000:
        print(f"  Subsampling to 500K for IF training (full set = {len(embeddings):,})...")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(embeddings), 500_000, replace=False)
        iso_forest.fit(embeddings[idx])
    else:
        iso_forest.fit(embeddings)

    if_path = os.path.join(args.output_dir, "isolation_forest.joblib")
    joblib.dump(iso_forest, if_path)
    elapsed_if = time.time() - t1
    print(f"  Trained in {elapsed_if:.1f}s")
    print(f"  Saved: {if_path} ({os.path.getsize(if_path) / 1e6:.1f} MB)")

    # Quick sanity check
    scores = iso_forest.score_samples(embeddings[:1000])
    print(f"  Sanity check (first 1000): score mean={scores.mean():.4f}, std={scores.std():.4f}")

    print("\n=== DONE ===")
    print(f"  reference_embeddings.npz: {emb_path}")
    print(f"  isolation_forest.joblib:  {if_path}")


if __name__ == "__main__":
    main()
