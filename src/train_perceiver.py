"""
Train Perceiver IO masked autoencoder on ZTF alert shards.

Self-supervised training: masks 50% of photometry tokens and reconstructs them.
No labels needed. Produces 128D embeddings comparable to RTF.

Usage:
    python train_perceiver.py \
        --data-dir /tmp/shards \
        --output-dir runs/perceiver \
        --epochs 25 \
        --batch-size 256

    # Resume from checkpoint:
    python train_perceiver.py \
        --data-dir /tmp/shards \
        --output-dir runs/perceiver \
        --resume-from runs/perceiver/checkpoint.pt
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from dataset import ShardedDataset, collate_fn
from perceiver_model import PerceiverAutoencoder


def train_one_epoch(model, loader, optimizer, epoch, total_epochs, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    metrics_sum = {}
    n_batches = 0
    t_start = time.time()

    for batch in loader:
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        images = batch["images"].to(device)

        # Forward (with masking)
        out = model(x, pad_mask, images)
        loss_dict = model.compute_loss(x, pad_mask, out)

        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val
        n_batches += 1

        if n_batches % 100 == 0:
            elapsed = time.time() - t_start
            avg_loss = metrics_sum.get("total_loss", 0) / n_batches
            avg_acc = metrics_sum.get("band_acc", 0) / n_batches
            rate = n_batches / elapsed if elapsed > 0 else 0
            print(
                f"    Epoch {epoch + 1}/{total_epochs} | batch {n_batches} | "
                f"loss={avg_loss:.4f} | band_acc={avg_acc:.3f} | "
                f"{rate:.1f} batch/s | {elapsed:.0f}s elapsed",
                flush=True,
            )

    return {k: v / max(n_batches, 1) for k, v in metrics_sum.items()}


def extract_embeddings(model, loader, device):
    """Extract 128D embeddings from all samples."""
    model.eval()
    all_emb, all_fine, all_coarse, all_ids = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            images = batch["images"].to(device)

            emb = model.embed(x, pad_mask, images)  # (B, 128)
            all_emb.append(emb.cpu())
            all_fine.append(batch["label_fine"])
            all_coarse.append(batch["label_coarse"])
            all_ids.extend(batch["obj_ids"])

    return {
        "embeddings": torch.cat(all_emb).numpy(),
        "labels_fine": torch.cat(all_fine).numpy(),
        "labels_coarse": torch.cat(all_coarse).numpy(),
        "obj_ids": np.array(all_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Perceiver IO masked autoencoder")
    parser.add_argument("--data-dir", required=True, help="Path to shards directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-latents", type=int, default=32)
    parser.add_argument("--n-cross", type=int, default=2)
    parser.add_argument("--n-self", type=int, default=6)
    parser.add_argument("--mask-ratio", type=float, default=0.50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--image-backbone", choices=["simple", "zoobot", "dinov2"], default="simple"
    )
    parser.add_argument("--resume-from", default=None, help="Path to checkpoint.pt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Dataset ---
    print(f"Loading shards from {args.data_dir}...")
    train_ds = ShardedDataset(args.data_dir, use_images=True)

    is_iterable = isinstance(train_ds, IterableDataset)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not is_iterable,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
        persistent_workers=False if is_iterable else (args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
    )

    # --- Model ---
    model = PerceiverAutoencoder(
        phot_channels=7,
        meta_channels=30,
        d_model=args.d_model,
        n_latents=args.n_latents,
        n_cross=args.n_cross,
        n_self=args.n_self,
        n_heads=8,
        dropout=0.1,
        embed_dim=args.embed_dim,
        mask_ratio=args.mask_ratio,
        image_backbone=args.image_backbone,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Resume
    start_epoch = 0
    best_loss = float("inf")
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 60}")
    print("Perceiver IO Masked Autoencoder Training")
    print(f"  Parameters: {n_params:,}")
    print(f"  Batch size: {args.batch_size}")
    print(
        f"  Latents: {args.n_latents} | Cross layers: {args.n_cross} | Self layers: {args.n_self}"
    )
    print(f"  Mask ratio: {args.mask_ratio}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Device: {args.device}")
    print(f"{'=' * 60}\n")

    # --- Training loop ---
    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_m = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            args.epochs,
            args.device,
        )
        scheduler.step()
        dt = time.time() - t0

        entry = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_m.items()},
        }
        history.append(entry)

        # Save best
        if train_m["total_loss"] < best_loss:
            best_loss = train_m["total_loss"]
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best_model.pt"),
            )

        # Save checkpoint
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "checkpoint.pt"))

        print(
            f"  [{epoch + 1:3d}/{args.epochs}] "
            f"loss={train_m['total_loss']:.4f} "
            f"recon_cont={train_m['recon_cont']:.4f} "
            f"recon_band={train_m['recon_band']:.4f} "
            f"band_acc={train_m['band_acc']:.3f} "
            f"[{dt:.1f}s]"
        )

    # --- Extract embeddings ---
    print("\nLoading best model for embedding extraction...")
    best_state = torch.load(
        os.path.join(args.output_dir, "best_model.pt"),
        weights_only=True,
        map_location=args.device,
    )
    model.load_state_dict(best_state)

    print("Extracting embeddings...")
    emb_data = extract_embeddings(model, train_loader, args.device)
    np.savez(
        os.path.join(args.output_dir, "embeddings.npz"),
        **emb_data,
    )
    print(f"  Saved embeddings: {emb_data['embeddings'].shape}")

    # Save summary
    summary = {
        "model": "PerceiverIO",
        "embed_dim": args.embed_dim,
        "d_model": args.d_model,
        "n_latents": args.n_latents,
        "n_cross": args.n_cross,
        "n_self": args.n_self,
        "mask_ratio": args.mask_ratio,
        "n_params": n_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_loss": best_loss,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\n{'=' * 60}")
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
