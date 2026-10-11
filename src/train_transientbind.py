"""
Train TransientBind: cross-modal contrastive learning for ZTF alerts.

Trains a three-branch encoder (light curve + image + metadata) using
symmetric InfoNCE loss on the 1.74M unlabeled alert archive.

Usage:
    python train_transientbind.py \
        --data-dir /tmp/shards \
        --output-dir runs/transientbind \
        --epochs 25 \
        --batch-size 512

    # Resume from checkpoint:
    python train_transientbind.py \
        --data-dir /tmp/shards \
        --output-dir runs/transientbind \
        --resume-from runs/transientbind/checkpoint.pt \
        --epochs 25
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from dataset import ShardedDataset, collate_fn
from transientbind_model import TransientBind


def train_one_epoch(
    model, loader, optimizer, epoch, total_epochs, device, grad_clip=1.0, grad_accum_steps=1
):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    metrics_sum = {}
    n_batches = 0
    t_start = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        images = batch["images"].to(device)

        # Forward with augmentation
        lc_emb, img_emb, meta_emb = model.forward_augmented(x, pad_mask, images)
        loss_dict = model.compute_contrastive_loss(lc_emb, img_emb, meta_emb)

        # Scale loss for gradient accumulation
        loss = loss_dict["total_loss"] / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val
        n_batches += 1

        # Progress every 100 batches
        if n_batches % 100 == 0:
            elapsed = time.time() - t_start
            avg_loss = metrics_sum.get("total_loss", 0) / n_batches
            rate = n_batches / elapsed if elapsed > 0 else 0
            print(
                f"    Epoch {epoch + 1}/{total_epochs} | batch {n_batches} | "
                f"loss={avg_loss:.4f} | "
                f"acc_lc_img={metrics_sum.get('acc_lc_img', 0) / n_batches:.3f} | "
                f"{rate:.1f} batch/s | {elapsed:.0f}s elapsed",
                flush=True,
            )

    return {k: v / max(n_batches, 1) for k, v in metrics_sum.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on validation set."""
    model.eval()
    metrics_sum = {}
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        images = batch["images"].to(device)

        # No augmentation during eval
        lc_emb, img_emb, meta_emb = model(x, pad_mask, images)
        loss_dict = model.compute_contrastive_loss(lc_emb, img_emb, meta_emb)

        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in metrics_sum.items()}


def extract_embeddings(model, loader, device):
    """Extract fused 128D embeddings from all samples."""
    model.eval()
    all_emb, all_fine, all_coarse, all_ids = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            images = batch["images"].to(device)

            emb = model.embed(x, pad_mask, images)  # (B, 128)
            model.reconstruction_error(x, pad_mask, images)  # (B,)

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
    parser = argparse.ArgumentParser(description="Train TransientBind contrastive model")
    parser.add_argument("--data-dir", required=True, help="Path to shards directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Linear LR warmup epochs (critical for contrastive)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--image-backbone", choices=["simple", "zoobot", "dinov2"], default="simple"
    )
    parser.add_argument("--resume-from", default=None, help="Path to checkpoint.pt to resume")
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
        drop_last=True,  # important: contrastive needs consistent batch size
    )

    # --- Model ---
    # Determine metadata channels from data format
    # ShardedDataset produces x of shape (B, L, 37): 7 phot + 30 meta
    meta_in = 30  # 37 total - 7 photometry

    model = TransientBind(
        lc_in_channels=7,
        meta_in_features=meta_in,
        d_model=args.d_model,
        embed_dim=args.embed_dim,
        n_heads=8,
        enc_layers=args.enc_layers,
        d_ff=args.d_model * 2,
        dropout=0.1,
        image_backbone=args.image_backbone,
        temperature=args.temperature,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Warmup + cosine decay: linear warmup for first N epochs, then cosine
    warmup_epochs = args.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear 0 → 1
        # Cosine decay from 1 → 0.01 over remaining epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        import math

        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
    effective_batch = args.batch_size * args.grad_accum

    print(f"\n{'=' * 60}")
    print("TransientBind Training (v2 — with projection heads)")
    print(f"  Parameters: {n_params:,}")
    print(
        f"  Batch size: {args.batch_size} × {args.grad_accum} accum = {effective_batch} effective"
    )
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Temperature: {args.temperature}")
    print(f"  LR warmup: {warmup_epochs} epochs")
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
            grad_accum_steps=args.grad_accum,
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

        # Save checkpoint for resume
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
            f"lc↔img={train_m['loss_lc_img']:.4f} "
            f"lc↔meta={train_m['loss_lc_meta']:.4f} "
            f"img↔meta={train_m['loss_img_meta']:.4f} "
            f"acc_lc_img={train_m['acc_lc_img']:.3f} "
            f"temp={train_m['temperature']:.4f} "
            f"[{dt:.1f}s]"
        )

    # --- Extract embeddings with best model ---
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
        "model": "TransientBind",
        "embed_dim": args.embed_dim,
        "d_model": args.d_model,
        "enc_layers": args.enc_layers,
        "temperature": args.temperature,
        "n_params": n_params,
        "epochs": args.epochs,
        "batch_size": effective_batch,
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
