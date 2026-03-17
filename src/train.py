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
from torch.utils.data import DataLoader

from dataset import AlertDataset, MetaNPZDataset, PhotoNPZDataset, collate_fn
from model import LightCurveCompressor


def beta_schedule(epoch, total_epochs, target_beta, warmup_frac=0.2):
    warmup = int(total_epochs * warmup_frac)
    if epoch < warmup:
        return target_beta * epoch / warmup
    return target_beta


def train_one_epoch(
    model, loader, optimizer, epoch, total_epochs, target_beta, device, grad_clip=1.0
):
    model.train()
    beta = (
        beta_schedule(epoch, total_epochs, target_beta) if model.mode == "vae" else 0.0
    )

    metrics_sum = {}
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        out = model(x, pad_mask)
        loss_dict = model.compute_loss(x, pad_mask, out, beta)

        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val
        n_batches += 1

    return {k: v / n_batches for k, v in metrics_sum.items()}


@torch.no_grad()
def evaluate(model, loader, epoch, total_epochs, target_beta, device):
    model.eval()
    beta = (
        beta_schedule(epoch, total_epochs, target_beta) if model.mode == "vae" else 0.0
    )

    metrics_sum = {}
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        out = model(x, pad_mask)
        loss_dict = model.compute_loss(x, pad_mask, out, beta)

        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            metrics_sum[k] = metrics_sum.get(k, 0.0) + val
        n_batches += 1

    return {k: v / n_batches for k, v in metrics_sum.items()}


def extract_embeddings(model, loader, device):
    """Extract latent embeddings from all samples."""
    model.eval()
    all_emb, all_fine, all_coarse, all_recon, all_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            emb = model.embed(x, pad_mask)
            recon_err = model.reconstruction_error(x, pad_mask)
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
):
    suffix = "_meta" if use_metadata else ""
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
        train_ds = MetaNPZDataset(os.path.join(data_dir, "train"))
        val_ds = MetaNPZDataset(os.path.join(data_dir, "val"))
        test_ds = MetaNPZDataset(os.path.join(data_dir, "test"))
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

    in_channels = train_ds.in_channels

    pin = device != "cpu"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
    )

    model = LightCurveCompressor(
        mode=mode,
        latent_dim=latent_dim,
        in_channels=in_channels,
        d_model=d_model,
        num_codes=num_codes,
        commitment_cost=commitment_cost,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    n_params = sum(p.numel() for p in model.parameters())
    comp = model.compression_info()
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
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, epoch, epochs, target_beta, device
        )
        val_m = evaluate(model, val_loader, epoch, epochs, target_beta, device)
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
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

        if epoch % 10 == 0 or epoch == epochs - 1:
            extra = ""
            if mode == "vae":
                extra = f"kld={val_m['kld']:.4f} beta={val_m['beta']:.3f} "
            elif mode == "vqvae":
                extra = f"vq={val_m['vq_loss']:.4f} cb_use={val_m.get('codebook_usage', 0):.0f} "
            print(
                f"  [{epoch:3d}/{epochs}] "
                f"train={train_m['total_loss']:.4f} "
                f"val={val_m['total_loss']:.4f} "
                f"(cont={val_m['recon_cont']:.4f} band={val_m['recon_band']:.4f}) "
                f"{extra}"
                f"band_acc={val_m['band_acc']:.3f} "
                f"[{dt:.1f}s]"
            )

    # Best model → test evaluation
    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"), weights_only=True)
    )
    test_m = evaluate(model, test_loader, best_epoch, epochs, target_beta, device)

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
