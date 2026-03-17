"""
Evaluate reconstruction quality in physical units (magnitudes, days).

Denormalizes model outputs and computes:
  - Per-band magnitude residuals (original vs reconstructed)
  - Time reconstruction errors (days)
  - Reconstructed light curve plots (original vs decoded)
  - Per-class reconstruction quality

Usage:
    python src/evaluate_physical.py \
        --runs-dir runs \
        --data-dir /path/to/photo_events \
        --output-dir analysis/physical
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PhotoNPZDataset, collate_fn
from model import LightCurveCompressor

COARSE_NAMES = {0: "SNIa", 1: "SNcc", 2: "Cataclysmic", 3: "AGN", 4: "TDE"}
FINE_NAMES = {
    0: "SN Ia", 1: "SN Ib", 2: "SN Ic", 3: "SN II", 4: "SN IIP",
    5: "SN IIn", 6: "SN IIb", 7: "Cataclysmic", 8: "AGN", 9: "TDE",
}
BAND_NAMES = {0: "g", 1: "r", 2: "i"}
BAND_COLORS = {0: "#1f77b4", 1: "#d62728", 2: "#ff7f0e"}


def load_stats(data_dir):
    """Load normalization stats and return (mean, std) as flat 1D arrays."""
    path = os.path.join(data_dir, "feature_stats_day100.npz")
    st = np.load(path)
    return st["mean"].flatten().astype(np.float32), st["std"].flatten().astype(np.float32)


def denormalize_batch(x_norm, mean, std):
    """Denormalize continuous channels back to physical units.

    Args:
        x_norm: (B, L, 7) normalized tensor
        mean, std: (4,) arrays for [log1p_dt, log1p_dt_prev, logflux, logflux_err]

    Returns:
        dt_days: (B, L) time since first detection in days
        logflux: (B, L) log10(flux)
        logflux_err: (B, L) log10(flux) uncertainty
        band: (B, L) integer band index
    """
    cont = x_norm[..., :4].numpy() if isinstance(x_norm, torch.Tensor) else x_norm[..., :4]
    # Undo normalization
    cont_phys = cont * std + mean
    # Undo log1p for time channels
    dt_days = np.expm1(cont_phys[..., 0])
    logflux = cont_phys[..., 2]
    logflux_err = cont_phys[..., 3]
    # Convert logflux to magnitude: mag = -2.5 * log10(flux) = -2.5 * logflux
    mag = -2.5 * logflux
    mag_err = 2.5 * logflux_err  # error propagation (approx)

    band_oh = x_norm[..., 4:7]
    if isinstance(band_oh, torch.Tensor):
        band_oh = band_oh.numpy()
    band = band_oh.argmax(axis=-1)

    return dt_days, mag, mag_err, band


@torch.no_grad()
def compute_physical_metrics(model, loader, mean, std, device):
    """Compute per-sample physical-unit reconstruction errors."""
    model.eval()
    all_results = []

    for batch in loader:
        x = batch["x"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        labels_fine = batch["label_fine"].numpy()
        labels_coarse = batch["label_coarse"].numpy()

        # Reconstruct
        z = model.embed(x, pad_mask)
        cont_hat, band_logits = model.decode(z)
        L = x.shape[1]
        cont_hat = cont_hat[:, :L]
        band_logits = band_logits[:, :L]

        # Build reconstructed tensor in same format as input
        band_pred_oh = torch.nn.functional.one_hot(band_logits.argmax(-1), 3).float()
        x_recon = torch.cat([cont_hat, band_pred_oh], dim=-1).cpu()
        x_orig = x.cpu()
        valid = ~pad_mask.cpu()

        # Denormalize both
        dt_orig, mag_orig, magerr_orig, band_orig = denormalize_batch(x_orig, mean, std)
        dt_recon, mag_recon, magerr_recon, band_recon = denormalize_batch(x_recon, mean, std)

        B = x.shape[0]
        for i in range(B):
            v = valid[i].numpy()
            n = v.sum()
            if n == 0:
                continue

            mag_residual = np.abs(mag_orig[i][v] - mag_recon[i][v])
            dt_residual = np.abs(dt_orig[i][v] - dt_recon[i][v])
            band_correct = (band_orig[i][v] == band_recon[i][v]).mean()

            # Per-band magnitude residuals
            per_band_mag = {}
            for b_id in range(3):
                mask = band_orig[i][v] == b_id
                if mask.sum() > 0:
                    per_band_mag[BAND_NAMES[b_id]] = float(mag_residual[mask].mean())

            all_results.append({
                "obj_id": batch["obj_ids"][i],
                "label_fine": int(labels_fine[i]),
                "label_coarse": int(labels_coarse[i]),
                "n_obs": int(n),
                "mag_residual_mean": float(mag_residual.mean()),
                "mag_residual_median": float(np.median(mag_residual)),
                "mag_residual_90pct": float(np.percentile(mag_residual, 90)),
                "dt_residual_mean_days": float(dt_residual.mean()),
                "band_accuracy": float(band_correct),
                "per_band_mag": per_band_mag,
            })

    return all_results


def plot_reconstructed_lightcurves(model, loader, mean, std, device, output_dir,
                                   n_per_class=3):
    """Plot original vs reconstructed light curves for sample sources."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Collect samples per coarse class
    samples_by_class = {k: [] for k in COARSE_NAMES}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            z = model.embed(x, pad_mask)
            cont_hat, band_logits = model.decode(z)
            L = x.shape[1]
            cont_hat = cont_hat[:, :L]
            band_logits = band_logits[:, :L]

            band_pred_oh = torch.nn.functional.one_hot(band_logits.argmax(-1), 3).float()
            x_recon = torch.cat([cont_hat, band_pred_oh], dim=-1).cpu()
            x_orig = x.cpu()
            valid = ~pad_mask.cpu()

            for i in range(x.shape[0]):
                cls = int(batch["label_coarse"][i])
                if len(samples_by_class[cls]) >= n_per_class:
                    continue
                samples_by_class[cls].append({
                    "obj_id": batch["obj_ids"][i],
                    "x_orig": x_orig[i],
                    "x_recon": x_recon[i],
                    "valid": valid[i],
                    "label_fine": int(batch["label_fine"][i]),
                })

            if all(len(v) >= n_per_class for v in samples_by_class.values()):
                break

    # Plot
    for cls, samples in samples_by_class.items():
        for s in samples:
            v = s["valid"].numpy()
            dt_o, mag_o, _, band_o = denormalize_batch(
                s["x_orig"].unsqueeze(0), mean, std
            )
            dt_r, mag_r, _, band_r = denormalize_batch(
                s["x_recon"].unsqueeze(0), mean, std
            )
            dt_o, mag_o, band_o = dt_o[0][v], mag_o[0][v], band_o[0][v]
            dt_r, mag_r, band_r = dt_r[0][v], mag_r[0][v], band_r[0][v]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

            # Original
            for b_id in range(3):
                mask = band_o == b_id
                if mask.any():
                    ax1.scatter(dt_o[mask], mag_o[mask], c=BAND_COLORS[b_id],
                                s=15, label=f"ZTF-{BAND_NAMES[b_id]}", alpha=0.8)
            ax1.invert_yaxis()
            ax1.set_xlabel("Days since first detection")
            ax1.set_ylabel("Magnitude")
            ax1.set_title(f"Original: {s['obj_id']}")
            ax1.legend(fontsize=8)

            # Reconstructed
            for b_id in range(3):
                mask = band_r == b_id
                if mask.any():
                    ax2.scatter(dt_r[mask], mag_r[mask], c=BAND_COLORS[b_id],
                                s=15, label=f"ZTF-{BAND_NAMES[b_id]}", alpha=0.8)
            ax2.invert_yaxis()
            ax2.set_xlabel("Days since first detection")
            ax2.set_title(f"Reconstructed ({FINE_NAMES.get(s['label_fine'], '?')})")
            ax2.legend(fontsize=8)

            plt.tight_layout()
            fname = f"{COARSE_NAMES[cls]}_{s['obj_id']}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"  Saved {sum(len(v) for v in samples_by_class.values())} light curve plots")


def summarize_results(results, output_dir):
    """Aggregate and print physical-unit metrics."""
    os.makedirs(output_dir, exist_ok=True)

    mag_res = [r["mag_residual_mean"] for r in results]
    mag_med = [r["mag_residual_median"] for r in results]
    mag_90 = [r["mag_residual_90pct"] for r in results]
    dt_res = [r["dt_residual_mean_days"] for r in results]
    band_acc = [r["band_accuracy"] for r in results]

    summary = {
        "n_sources": len(results),
        "mag_residual_mean": float(np.mean(mag_res)),
        "mag_residual_median": float(np.median(mag_med)),
        "mag_residual_90pct": float(np.mean(mag_90)),
        "dt_residual_mean_days": float(np.mean(dt_res)),
        "band_accuracy": float(np.mean(band_acc)),
    }

    # Per-class
    per_class = {}
    for cls_id, cls_name in COARSE_NAMES.items():
        cls_results = [r for r in results if r["label_coarse"] == cls_id]
        if not cls_results:
            continue
        per_class[cls_name] = {
            "n": len(cls_results),
            "mag_residual_mean": float(np.mean([r["mag_residual_mean"] for r in cls_results])),
            "mag_residual_median": float(np.median([r["mag_residual_median"] for r in cls_results])),
            "band_accuracy": float(np.mean([r["band_accuracy"] for r in cls_results])),
        }
    summary["per_class"] = per_class

    # Per-band
    per_band = {}
    for band_name in ["g", "r", "i"]:
        vals = [r["per_band_mag"].get(band_name) for r in results if band_name in r["per_band_mag"]]
        if vals:
            per_band[band_name] = {
                "mag_residual_mean": float(np.mean(vals)),
                "n_sources_with_band": len(vals),
            }
    summary["per_band"] = per_band

    with open(os.path.join(output_dir, "physical_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Overall ({len(results)} sources):")
    print(f"    Mag residual:  mean={summary['mag_residual_mean']:.3f}  "
          f"median={summary['mag_residual_median']:.3f}  "
          f"90th pct={summary['mag_residual_90pct']:.3f}")
    print(f"    Time residual: {summary['dt_residual_mean_days']:.2f} days")
    print(f"    Band accuracy: {summary['band_accuracy']:.1%}")

    print("\n  Per class:")
    for cls_name, m in per_class.items():
        print(f"    {cls_name:15s} (n={m['n']:4d}): "
              f"mag_resid={m['mag_residual_mean']:.3f}  band_acc={m['band_accuracy']:.1%}")

    print("\n  Per band:")
    for band_name, m in per_band.items():
        print(f"    ZTF-{band_name}: mag_resid={m['mag_residual_mean']:.3f} "
              f"(n={m['n_sources_with_band']})")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Physical-unit reconstruction evaluation")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific run dirs to evaluate (default: all)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-per-class", type=int, default=3,
                        help="Number of light curve plots per class")
    args = parser.parse_args()

    mean, std = load_stats(args.data_dir)
    stats_path = os.path.join(args.data_dir, "feature_stats_day100.npz")

    test_ds = PhotoNPZDataset(os.path.join(args.data_dir, "test"), stats_path, horizon=100.0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=4,
                             pin_memory=(args.device != "cpu"))

    run_dirs = args.models or sorted([
        d for d in os.listdir(args.runs_dir)
        if os.path.isdir(os.path.join(args.runs_dir, d))
    ])

    all_summaries = []
    for d in run_dirs:
        run_path = os.path.join(args.runs_dir, d) if "/" not in d else d
        summary_path = os.path.join(run_path, "summary.json")
        model_path = os.path.join(run_path, "best_model.pt")
        if not os.path.exists(model_path):
            continue

        with open(summary_path) as f:
            run_summary = json.load(f)

        mode = run_summary["mode"]
        latent_dim = run_summary["latent_dim"]
        print(f"\n{'='*60}")
        print(f"Evaluating {d}: {mode} dim={latent_dim}")
        print(f"{'='*60}")

        model = LightCurveCompressor(
            mode=mode, latent_dim=latent_dim,
            num_codes=run_summary.get("num_codes", 512),
        ).to(args.device)
        model.load_state_dict(
            torch.load(model_path, map_location=args.device, weights_only=True)
        )

        out_subdir = os.path.join(args.output_dir, d)

        # Physical metrics
        results = compute_physical_metrics(model, test_loader, mean, std, args.device)
        phys_summary = summarize_results(results, out_subdir)
        phys_summary["run"] = d
        phys_summary["mode"] = mode
        phys_summary["latent_dim"] = latent_dim
        all_summaries.append(phys_summary)

        # Light curve plots
        plot_reconstructed_lightcurves(
            model, test_loader, mean, std, args.device,
            os.path.join(out_subdir, "lightcurves"),
            n_per_class=args.plot_per_class,
        )

    # Combined summary
    with open(os.path.join(args.output_dir, "all_physical_metrics.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*80}")
    print("PHYSICAL METRICS SUMMARY")
    print(f"{'='*80}")
    print(f"{'model':>15} {'dim':>5} {'mag_mean':>8} {'mag_med':>8} {'mag_90':>8} "
          f"{'dt_days':>8} {'band%':>6}")
    print("-" * 65)
    for s in sorted(all_summaries, key=lambda x: (x["mode"], x["latent_dim"])):
        print(f"{s['mode']:>15} {s['latent_dim']:5d} "
              f"{s['mag_residual_mean']:8.3f} {s['mag_residual_median']:8.3f} "
              f"{s['mag_residual_90pct']:8.3f} {s['dt_residual_mean_days']:8.2f} "
              f"{s['band_accuracy']:6.1%}")


if __name__ == "__main__":
    main()
