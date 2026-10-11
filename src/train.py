"""
Train light curve compression models (AE / VAE / VQ-VAE) with latent dim sweep.

Usage:
    # Photometry only (from pre-processed NPZ):
    python src/train.py \
        --data-dir /fred/oz480/mcoughli/AppleCider/photo_events \
        --output-dir runs \
        --mode ae \
        --latent-dims 64 256

    # Photometry + alert metadata (from raw alerts.npy):
    python src/train.py \
        --alert-dir /fred/oz480/mcoughli/data_ztf \
        --splits /fred/oz480/mcoughli/AppleCider/photo_events/splits.csv \
        --labels-dir /fred/oz480/mcoughli/AppleCider/photo_events \
        --output-dir runs \
        --mode ae \
        --use-metadata \
        --latent-dims 64 256
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, IterableDataset

from dataset import (
    AlertDataset,
    MetaNPZDataset,
    PhotoNPZDataset,
    ShardedDataset,
    collate_fn,
)
from model import LightCurveCompressor


def _get_extra_args(batch, device):
    """Extract optional tensors (images, GP features) from batch."""
    images = batch["images"].to(device) if "images" in batch else None
    gp_features = batch["gp_features"].to(device) if "gp_features" in batch else None
    return images, gp_features


def beta_schedule(epoch, total_epochs, target_beta, warmup_frac=0.2):
    warmup = int(total_epochs * warmup_frac)
    if epoch < warmup:
        return target_beta * epoch / warmup
    return target_beta


def train_one_epoch(
    model,
    loader,
    optimizer,
    epoch,
    total_epochs,
    target_beta,
    device,
    grad_clip=1.0,
    cls_weight=0.0,
    band_weight=1.0,
    color_weight=0.0,
    gp_weight=0.0,
):
    model.train()
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    beta = (
        beta_schedule(epoch, total_epochs, target_beta) if core_model.mode == "vae" else 0.0
    )

    metrics_sum = {}
    n_batches = 0
    t_start = time.time()

    for batch in loader:
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        images, gp_features = _get_extra_args(batch, device)
        gp_targets = batch["gp_features"].to(device) if "gp_features" in batch and gp_weight > 0 else None

        labels = batch["label_coarse"].to(device) if cls_weight > 0 else None
        out = model(x, pad_mask, images, gp_features=gp_features)
        loss_dict = core_model.compute_loss(
            x, pad_mask, out, beta, labels=labels, cls_weight=cls_weight, band_weight=band_weight, color_weight=color_weight,
            gp_weight=gp_weight, gp_targets=gp_targets
        )

        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val
        n_batches += 1

        # Print progress every 200 batches so user knows it's alive
        if n_batches % 200 == 0:
            elapsed = time.time() - t_start
            avg_loss = metrics_sum.get('total_loss', 0) / n_batches
            rate = n_batches / elapsed if elapsed > 0 else 0
            print(f"    Epoch {epoch+1}/{total_epochs} | batch {n_batches} | "
                  f"loss={avg_loss:.4f} | {rate:.1f} batch/s | "
                  f"{elapsed:.0f}s elapsed", flush=True)

    return {k: v / n_batches for k, v in metrics_sum.items()}


@torch.no_grad()
def evaluate(model, loader, epoch, total_epochs, target_beta, device, cls_weight=0.0, band_weight=1.0, color_weight=0.0, gp_weight=0.0):
    model.eval()
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    beta = (
        beta_schedule(epoch, total_epochs, target_beta) if core_model.mode == "vae" else 0.0
    )

    metrics_sum = {}
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        images, gp_features = _get_extra_args(batch, device)
        gp_targets = batch["gp_features"].to(device) if "gp_features" in batch and gp_weight > 0 else None

        out = model(x, pad_mask, images, gp_features=gp_features)
        loss_dict = core_model.compute_loss(x, pad_mask, out, beta, band_weight=band_weight, color_weight=color_weight,
            gp_weight=gp_weight, gp_targets=gp_targets)

        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val
        n_batches += 1

    return {k: v / n_batches for k, v in metrics_sum.items()}


def extract_embeddings(model, loader, device):
    """Extract latent embeddings from all samples."""
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    core_model.eval()
    all_emb, all_fine, all_coarse, all_recon, all_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            images, gp_features = _get_extra_args(batch, device)
            emb = core_model.embed(x, pad_mask, images, gp_features=gp_features)
            recon_err = core_model.reconstruction_error(
                x, pad_mask, images, gp_features=gp_features
            )
            all_emb.append(emb.cpu())
            all_recon.append(recon_err.cpu())
            all_fine.append(batch["label_fine"])
            all_coarse.append(batch["label_coarse"])
            all_ids.extend(batch["obj_ids"])
    return {
        "embeddings": torch.cat(all_emb).numpy(),
        "recon_errors": torch.cat(all_recon).numpy(),
        "labels_fine": torch.cat(all_fine).numpy(),
        "labels_coarse": torch.cat(all_coarse).numpy(),
        "obj_ids": np.array(all_ids),
    }


def train_model(
    mode,
    latent_dim,
    data_dir,
    output_dir,
    epochs=200,
    batch_size=128,
    lr=1e-4,
    target_beta=1.0,
    d_model=128,
    horizon=100.0,
    device="cuda",
    num_workers=4,
    num_codes=512,
    commitment_cost=0.25,
    alert_dir=None,
    splits_path=None,
    labels_dir=None,
    use_metadata=False,
    use_images=False,
    image_backbone="simple",
    finetune_backbone=False,
    use_gp=False,
    gp_dim=114,
    max_detections=None,
    random_truncate=False,
    min_detections=3,
    cls_weight=0.0,
    num_classes=5,
    synthetic_dir=None,
    resume_from=None,
    band_weight=1.0,
    color_weight=0.0,
    gp_weight=0.0,
):
    suffix = ""
    if use_metadata:
        suffix += "_meta"
    if use_images:
        suffix += "_img"
        if finetune_backbone:
            suffix += "_ft"
    if use_gp:
        suffix += "_gp"
    if random_truncate:
        suffix += "_randtrunc"
    elif max_detections is not None:
        suffix += f"_n{max_detections}"
    if cls_weight > 0:
        suffix += f"_cls{cls_weight}"
    run_dir = os.path.join(output_dir, f"{mode}_dim{latent_dim}{suffix}")
    os.makedirs(run_dir, exist_ok=True)

    # Choose dataset:
    #   - AlertDataset: raw alerts.npy (slow, for testing)
    #   - MetaNPZDataset: pre-processed NPZ with metadata (fast, from preprocess_alerts.py)
    #   - PhotoNPZDataset: legacy NPZ with photometry only
    if alert_dir is not None:
        train_ds = AlertDataset(
            alert_dir,
            splits_path,
            "train",
            labels_dir,
            use_metadata=use_metadata,
            horizon=horizon,
        )
        val_ds = AlertDataset(
            alert_dir,
            splits_path,
            "val",
            labels_dir,
            use_metadata=use_metadata,
            horizon=horizon,
        )
        test_ds = AlertDataset(
            alert_dir,
            splits_path,
            "test",
            labels_dir,
            use_metadata=use_metadata,
            horizon=horizon,
        )
    elif use_metadata:
        train_dsets, val_dsets, test_dsets = [], [], []
        for d in data_dir.split(","):
            d = d.strip()
            # Fast path: check if directory contains packed shard files
            from pathlib import Path as _Path
            shard_files = sorted(_Path(d).glob("shard_*.pt")) if _Path(d).is_dir() else []
            if shard_files:
                print(f"  [SHARDS] Found {len(shard_files)} shard files in {d}")
                train_dsets.append(ShardedDataset(d, use_images=use_images, use_gp=use_gp, gp_dim=gp_dim, max_detections=max_detections, random_truncate=random_truncate, min_detections=min_detections))
                val_dsets.append(ShardedDataset(d, use_images=use_images, use_gp=use_gp, gp_dim=gp_dim, max_detections=max_detections))
                test_dsets.append(ShardedDataset(d, use_images=use_images, use_gp=use_gp, gp_dim=gp_dim, max_detections=max_detections))
            elif os.path.exists(os.path.join(d, "infer")):
                t_path = v_path = te_path = os.path.join(d, "infer")
                train_dsets.append(MetaNPZDataset(t_path, use_images=use_images, use_gp=use_gp, max_detections=max_detections, random_truncate=random_truncate, min_detections=min_detections))
                val_dsets.append(MetaNPZDataset(v_path, use_images=use_images, use_gp=use_gp, max_detections=max_detections))
                test_dsets.append(MetaNPZDataset(te_path, use_images=use_images, use_gp=use_gp, max_detections=max_detections))
            else:
                t_path = os.path.join(d, "train") if os.path.exists(os.path.join(d, "train")) else d
                v_path = os.path.join(d, "val") if os.path.exists(os.path.join(d, "val")) else d
                te_path = os.path.join(d, "test") if os.path.exists(os.path.join(d, "test")) else d
                train_dsets.append(MetaNPZDataset(t_path, use_images=use_images, use_gp=use_gp, max_detections=max_detections, random_truncate=random_truncate, min_detections=min_detections))
                val_dsets.append(MetaNPZDataset(v_path, use_images=use_images, use_gp=use_gp, max_detections=max_detections))
                test_dsets.append(MetaNPZDataset(te_path, use_images=use_images, use_gp=use_gp, max_detections=max_detections))
        
        train_ds = ConcatDataset(train_dsets) if len(train_dsets) > 1 else train_dsets[0]
        val_ds = ConcatDataset(val_dsets) if len(val_dsets) > 1 else val_dsets[0]
        test_ds = ConcatDataset(test_dsets) if len(test_dsets) > 1 else test_dsets[0]
    else:
        stats_path = os.path.join(data_dir, "feature_stats_day100.npz")
        if not os.path.exists(stats_path):
            stats_path = None
        train_ds = PhotoNPZDataset(
            os.path.join(data_dir, "train"), stats_path, horizon=horizon
        )
        val_ds = PhotoNPZDataset(
            os.path.join(data_dir, "val"), stats_path, horizon=horizon
        )
        test_ds = PhotoNPZDataset(
            os.path.join(data_dir, "test"), stats_path, horizon=horizon
        )

    in_channels = train_ds.datasets[0].in_channels if isinstance(train_ds, ConcatDataset) else train_ds.in_channels

    # Optionally merge synthetic data into training set
    if synthetic_dir is not None and os.path.isdir(
        os.path.join(synthetic_dir, "train")
    ):
        syn_ds = MetaNPZDataset(
            os.path.join(synthetic_dir, "train"),
            use_images=use_images,
            use_gp=use_gp,
            random_truncate=random_truncate,
            min_detections=min_detections,
        )
        print(f"  Adding {len(syn_ds)} synthetic sources to training set")
        train_ds = ConcatDataset([train_ds, syn_ds])

    pin = device != "cpu"
    pers_workers = num_workers > 0

    # IterableDataset (ShardedDataset) handles its own shuffling
    is_iterable = isinstance(train_ds, IterableDataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not is_iterable,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=False if is_iterable else pers_workers,
        prefetch_factor=2 if (num_workers > 0 and is_iterable) else (8 if num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pers_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pers_workers,
    )

    model = LightCurveCompressor(
        mode=mode,
        latent_dim=latent_dim,
        in_channels=in_channels,
        d_model=d_model,
        num_codes=num_codes,
        commitment_cost=commitment_cost,
        use_images=use_images,
        image_backbone=image_backbone,
        freeze_backbone=not finetune_backbone if use_images else None,
        gp_dim=gp_dim if use_gp else 0,
        num_classes=num_classes if cls_weight > 0 else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    n_params = sum(p.numel() for p in model.parameters())
    
    if resume_from is not None:
        ckpt_path = resume_from
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                # Full checkpoint with optimizer/scheduler/epoch
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                best_epoch = ckpt.get("best_epoch", 0)
                print(f"  Resuming from epoch {start_epoch} (best_val_loss={best_val_loss:.6f})")
            else:
                # Legacy: raw state dict only
                model.load_state_dict(ckpt)
                start_epoch = 0
                best_val_loss = float("inf")
                best_epoch = 0
                print("  Loaded model weights (no epoch info, starting from epoch 0)")
        else:
            print(f"  Warning: {ckpt_path} not found, training from scratch")
            start_epoch = 0
            best_val_loss = float("inf")
            best_epoch = 0
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        best_epoch = 0

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)

    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    comp = core_model.compression_info()
    print(f"\n{'=' * 60}")
    print(
        f"Training {mode.upper()}: latent_dim={latent_dim}, in_channels={in_channels}, d_model={d_model}"
    )
    print(f"Parameters: {n_params:,}")
    print(
        f"Compression: {comp['compression_ratio']:.1f}x ({comp['compressed_bytes']}B/alert)"
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"{'=' * 60}")

    history = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_m = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            epochs,
            target_beta,
            device,
            cls_weight=cls_weight,
            band_weight=band_weight,
            color_weight=color_weight,
            gp_weight=gp_weight,
        )
        val_m = evaluate(
            model, val_loader, epoch, epochs, target_beta, device,
            cls_weight=cls_weight, band_weight=band_weight, color_weight=color_weight,
            gp_weight=gp_weight,
        )
        scheduler.step()
        dt = time.time() - t0

        entry = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(entry)

        if val_m["total_loss"] < best_val_loss:
            best_val_loss = val_m["total_loss"]
            best_epoch = epoch
            state_dict_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict_to_save, os.path.join(run_dir, "best_model.pt"))

        # Save full checkpoint for resume on timeout
        ckpt = {
            "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
        torch.save(ckpt, os.path.join(run_dir, "checkpoint.pt"))

        if True: # Output every single epoch
            extra = ""
            if mode == "vae":
                extra = f"kld={val_m['kld']:.4f} beta={val_m['beta']:.3f} "
            elif mode == "vqvae":
                extra = f"vq={val_m['vq_loss']:.4f} cb_use={val_m.get('codebook_usage', 0):.0f} "
            color_str = f"color={val_m.get('color_loss', 0):.4f} " if color_weight > 0 else ""
            gp_str = f"gp={val_m.get('gp_loss', 0):.4f} " if gp_weight > 0 else ""
            print(
                f"  [{epoch:3d}/{epochs}] "
                f"train={train_m['total_loss']:.4f} "
                f"val={val_m['total_loss']:.4f} "
                f"(cont={val_m['recon_cont']:.4f} band={val_m['recon_band']:.4f}) "
                f"{color_str}"
                f"{gp_str}"
                f"{extra}"
                f"band_acc={val_m['band_acc']:.3f} "
                f"[{dt:.1f}s]"
            )

    # Best model → test evaluation
    best_state_dict = torch.load(os.path.join(run_dir, "best_model.pt"), weights_only=True, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(best_state_dict)
    else:
        model.load_state_dict(best_state_dict)
    test_m = evaluate(
        model,
        test_loader,
        best_epoch,
        epochs,
        target_beta,
        device,
        cls_weight=cls_weight,
        band_weight=band_weight,
        color_weight=color_weight,
    )

    # Extract embeddings for downstream evaluation
    test_data = extract_embeddings(model, test_loader, device)
    np.savez(os.path.join(run_dir, "test_embeddings.npz"), **test_data)

    train_data = extract_embeddings(model, train_loader, device)
    np.savez(os.path.join(run_dir, "train_embeddings.npz"), **train_data)

    # Save summary
    summary = {
        "mode": mode,
        "latent_dim": latent_dim,
        "in_channels": in_channels,
        "d_model": d_model,
        "use_metadata": use_metadata,
        "use_images": use_images,
        "use_gp": use_gp,
        "gp_dim": gp_dim if use_gp else 0,
        "max_detections": max_detections,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "target_beta": target_beta if mode == "vae" else None,
        "num_codes": num_codes if mode == "vqvae" else None,
        "test_metrics": test_m,
        **comp,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\n  Best epoch: {best_epoch}")
    print(
        f"  Test recon: cont={test_m['recon_cont']:.4f} band={test_m['recon_band']:.4f}"
    )
    print(f"  Test band_acc: {test_m['band_acc']:.3f}")
    print(f"  Saved to {run_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train light curve compression models")
    parser.add_argument(
        "--data-dir", default=None, help="Path to photo_events/ (for PhotoNPZDataset)"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["ae", "vae", "vqvae"], default="vae")
    parser.add_argument(
        "--latent-dims", nargs="+", type=int, default=[32, 64, 128, 256]
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight (VAE only)")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--horizon", type=float, default=100.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--num-codes", type=int, default=512, help="Codebook size (VQ-VAE only)"
    )
    parser.add_argument(
        "--commitment-cost", type=float, default=0.25, help="VQ commitment cost"
    )
    # Alert metadata options
    parser.add_argument(
        "--alert-dir",
        default=None,
        help="Path to data_ztf/ with raw alerts (enables AlertDataset)",
    )
    parser.add_argument(
        "--splits", default=None, help="Path to splits.csv (required with --alert-dir)"
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Path to dir with {split}/{obj_id}.npz labels (required with --alert-dir)",
    )
    parser.add_argument(
        "--use-metadata",
        action="store_true",
        help="Include alert metadata in encoder input",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Include cutout images in encoder input (requires preprocessed data with images)",
    )
    parser.add_argument(
        "--image-backbone",
        choices=["simple", "zoobot", "zoobot_large", "dinov2"],
        default="simple",
        help="Image tower backbone: simple (4-layer CNN), zoobot (ConvNeXt-pico), zoobot_large (ConvNeXt-large, ~198M params), or dinov2 (ViT-S/14, ~22M params)",
    )
    parser.add_argument(
        "--finetune-backbone",
        action="store_true",
        help="Fine-tune the pretrained image backbone (default: frozen)",
    )
    parser.add_argument(
        "--use-gp",
        action="store_true",
        help="Include GP features from lightcurve-fitting in encoder input",
    )
    parser.add_argument(
        "--gp-dim",
        type=int,
        default=114,
        help="Number of GP features (from extract_features output)",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="Truncate each source to first N observations (fixed, for evaluation)",
    )
    parser.add_argument(
        "--cls-weight",
        type=float,
        default=0.0,
        help="Weight for joint classification loss (0=reconstruction only)",
    )
    parser.add_argument(
        "--band-weight",
        type=float,
        default=1.0,
        help="Weight for band reconstruction loss (0=cont-only, 1=default 50/50)",
    )
    parser.add_argument(
        "--color-weight",
        type=float,
        default=0.0,
        help="Weight for color (g-r) prediction loss (0=disabled)",
    )
    parser.add_argument(
        "--gp-weight",
        type=float,
        default=0.0,
        help="Weight for GP feature prediction loss (0=disabled)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Number of classes for joint classification (default: 5 coarse)",
    )
    parser.add_argument(
        "--synthetic-dir",
        default=None,
        help="Path to synthetic data dir (from generate_synthetic.py). Merged into training set.",
    )
    parser.add_argument(
        "--random-truncate",
        action="store_true",
        help="Randomly truncate each source to [min-det, L] per epoch (variable-length training)",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=3,
        help="Minimum detections when using --random-truncate",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to .pt checkpoint to resume training from",
    )
    args = parser.parse_args()

    all_summaries = []
    for latent_dim in args.latent_dims:
        s = train_model(
            mode=args.mode,
            latent_dim=latent_dim,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            target_beta=args.beta,
            d_model=args.d_model,
            horizon=args.horizon,
            device=args.device,
            num_workers=args.num_workers,
            num_codes=args.num_codes,
            commitment_cost=args.commitment_cost,
            alert_dir=args.alert_dir,
            splits_path=args.splits,
            labels_dir=args.labels_dir,
            use_metadata=args.use_metadata,
            use_images=args.use_images,
            image_backbone=args.image_backbone,
            finetune_backbone=args.finetune_backbone,
            use_gp=args.use_gp,
            gp_dim=args.gp_dim,
            max_detections=args.max_detections,
            random_truncate=args.random_truncate,
            min_detections=args.min_detections,
            cls_weight=args.cls_weight,
            num_classes=args.num_classes,
            synthetic_dir=args.synthetic_dir,
            resume_from=args.resume_from,
            band_weight=args.band_weight,
            color_weight=args.color_weight,
            gp_weight=args.gp_weight,
        )
        all_summaries.append(s)

    with open(os.path.join(args.output_dir, f"sweep_{args.mode}.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"{args.mode.upper()} SWEEP RESULTS")
    print(f"{'=' * 80}")
    print(
        f"{'dim':>5} {'recon_cont':>10} {'recon_band':>10} {'band_acc':>9} {'compress':>8} {'bytes':>6}"
    )
    print("-" * 55)
    for s in all_summaries:
        tm = s["test_metrics"]
        print(
            f"{s['latent_dim']:5d} {tm['recon_cont']:10.4f} {tm['recon_band']:10.4f} "
            f"{tm['band_acc']:9.3f} {s['compression_ratio']:7.1f}x {s['compressed_bytes']:5d}B"
        )


if __name__ == "__main__":
    main()
