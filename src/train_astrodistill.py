"""
Train AstroDistill: Data2Vec-style self-distillation for ZTF alerts.

Trains a student encoder to predict a teacher's contextualized latent
representations at masked positions. The teacher is an exponential
moving average (EMA) of the student.

Usage:
    python train_astrodistill.py \
        --data-dir /tmp/shards \
        --output-dir runs/astrodistill \
        --epochs 25 \
        --batch-size 256

    # Resume from checkpoint:
    python train_astrodistill.py \
        --data-dir /tmp/shards \
        --output-dir runs/astrodistill \
        --resume-from runs/astrodistill/checkpoint.pt \
        --epochs 25
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from astrodistill_model import AstroDistill
from dataset import ShardedDataset, collate_fn


def cosine_ema_schedule(epoch: int, total_epochs: int, ema_start: float, ema_end: float) -> float:
    """Cosine schedule for EMA decay: starts at ema_start, ends at ema_end.

    Early training: low decay (teacher follows student closely).
    Late training: high decay (teacher is very stable).
    """
    progress = epoch / max(total_epochs - 1, 1)
    return ema_end - (ema_end - ema_start) * (math.cos(math.pi * progress) + 1) / 2


def train_one_epoch(
    model, loader, optimizer, epoch, total_epochs, device,
    ema_start, ema_end, grad_clip=1.0, grad_accum_steps=1,
):
    """Train for one epoch with EMA teacher updates."""
    model.train()
    loss_sum = 0.0
    n_batches = 0
    t_start = time.time()
    optimizer.zero_grad()

    # Compute EMA decay for this epoch
    ema_decay = cosine_ema_schedule(epoch, total_epochs, ema_start, ema_end)

    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        images = batch["images"].to(device)

        # Forward: teacher (no grad) + student (grad) + loss
        out = model(x, pad_mask, images)
        loss = out["total_loss"] / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            # Update teacher weights (EMA) after each optimizer step
            model.update_teacher(decay=ema_decay)

        loss_sum += out["loss_value"]
        n_batches += 1

        # Progress every 100 batches
        if n_batches % 100 == 0:
            elapsed = time.time() - t_start
            avg_loss = loss_sum / n_batches
            rate = n_batches / elapsed if elapsed > 0 else 0
            print(
                f"    Epoch {epoch+1}/{total_epochs} | batch {n_batches} | "
                f"loss={avg_loss:.4f} | ema={ema_decay:.4f} | "
                f"{rate:.1f} batch/s | {elapsed:.0f}s elapsed",
                flush=True,
            )

    avg_loss = loss_sum / max(n_batches, 1)
    return {
        "loss": avg_loss,
        "ema_decay": ema_decay,
    }


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract fused 128D embeddings from all samples."""
    model.eval()
    all_emb, all_fine, all_coarse, all_ids = [], [], [], []

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
    parser = argparse.ArgumentParser(description="Train AstroDistill self-distillation model")
    parser.add_argument("--data-dir", required=True, help="Path to shards directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="Initial EMA decay for teacher")
    parser.add_argument("--ema-end", type=float, default=0.9999,
                        help="Final EMA decay for teacher (cosine schedule)")
    parser.add_argument("--top-k-layers", type=int, default=2,
                        help="Average top-K transformer layers for teacher targets")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear LR warmup epochs")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-backbone", choices=["simple", "zoobot", "dinov2"], default="simple")
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
        drop_last=True,
    )

    # --- Model ---
    meta_in = 30  # 37 total - 7 photometry

    model = AstroDistill(
        phot_channels=7,
        meta_channels=meta_in,
        d_model=args.d_model,
        n_heads=8,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout=0.1,
        embed_dim=args.embed_dim,
        mask_ratio=args.mask_ratio,
        image_backbone=args.image_backbone,
        top_k_layers=args.top_k_layers,
        pred_hidden=args.d_model * 2,
        ema_decay=args.ema_decay,
    ).to(args.device)

    # Optimizer: only student and prediction MLP parameters (NOT teacher)
    trainable_params = [
        p for n, p in model.named_parameters()
        if "teacher_encoder" not in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

    # LR schedule: linear warmup + cosine decay
    warmup_epochs = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
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

    # Count parameters
    n_student = sum(p.numel() for n, p in model.named_parameters()
                    if "teacher_encoder" not in n)
    n_teacher = sum(p.numel() for n, p in model.named_parameters()
                    if "teacher_encoder" in n)
    n_total = sum(p.numel() for p in model.parameters())
    effective_batch = args.batch_size * args.grad_accum

    print(f"\n{'=' * 60}")
    print("AstroDistill Training (Data2Vec-style self-distillation)")
    print(f"  Parameters: {n_total:,} total ({n_student:,} student + {n_teacher:,} teacher)")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} accum = {effective_batch} effective")
    print(f"  Encoder: {args.n_layers} layers, d_model={args.d_model}")
    print(f"  Mask ratio: {args.mask_ratio}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  EMA: {args.ema_decay} → {args.ema_end} (cosine)")
    print(f"  Top-K layers: {args.top_k_layers}")
    print(f"  LR warmup: {warmup_epochs} epochs")
    print(f"  Device: {args.device}")
    print(f"{'=' * 60}\n")

    # --- Training loop ---
    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, epoch, args.epochs,
            args.device, args.ema_decay, args.ema_end,
            grad_accum_steps=args.grad_accum,
        )
        scheduler.step()
        dt = time.time() - t0

        entry = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_m["loss"],
            "ema_decay": train_m["ema_decay"],
        }
        history.append(entry)

        # Save best
        if train_m["loss"] < best_loss:
            best_loss = train_m["loss"]
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
            f"  [{epoch+1:3d}/{args.epochs}] "
            f"loss={train_m['loss']:.4f} "
            f"ema={train_m['ema_decay']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} "
            f"[{dt:.1f}s]"
        )

    # --- Extract embeddings with best model ---
    print("\nLoading best model for embedding extraction...")
    best_state = torch.load(
        os.path.join(args.output_dir, "best_model.pt"),
        weights_only=True, map_location=args.device,
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
        "model": "AstroDistill",
        "embed_dim": args.embed_dim,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "mask_ratio": args.mask_ratio,
        "ema_decay": args.ema_decay,
        "ema_end": args.ema_end,
        "top_k_layers": args.top_k_layers,
        "n_params_student": n_student,
        "n_params_total": n_total,
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
