"""
Generate synthetic light curves from lightcurve-fitting parametric models.

Samples parameters from fitted distributions (real ZTF data) and generates
synthetic photometry for data augmentation and "other" class training.

Models available:
  - Bazin, Villar, Tde, Arnett — represented in training data
  - Magnetar, ShockCooling, Afterglow, MetzgerKN — underrepresented or absent

Usage:
    python src/generate_synthetic.py \
        --fitting-dir /fred/oz480/mcoughli/AppleCider/fitting_results \
        --output-dir /fred/oz480/mcoughli/AppleCider/rtf/data_synthetic \
        --n-per-model 500 \
        --models Afterglow MetzgerKN Magnetar ShockCooling
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


# Model name → coarse label mapping
MODEL_TO_LABEL = {
    "Bazin": 0,  # SN Ia-like
    "Villar": 0,  # SN Ia-like
    "Tde": 4,  # TDE
    "Arnett": 1,  # SN cc-like
    "Magnetar": 1,  # SN cc-like (magnetar-powered SN)
    "ShockCooling": 1,  # SN cc-like (shock breakout)
    "Afterglow": 5,  # "Other" — GRB afterglow
    "MetzgerKN": 5,  # "Other" — kilonova
}

# ZTF-like observation properties
ZTF_BANDS = ["ztfg", "ztfr"]
ZTF_BAND_FIDS = [1, 2]
ZTF_MAG_LIMIT = 20.5
ZTF_MAG_ERR_FLOOR = 0.03


def load_param_distributions(fitting_dir, max_sources=500):
    """Load parameter distributions from real fitted sources."""

    params_by_model = {}
    files = sorted(os.listdir(fitting_dir))[:max_sources]
    for f in files:
        if not f.endswith(".json"):
            continue
        with open(os.path.join(fitting_dir, f)) as fh:
            data = json.load(fh)
        for band_result in data.get("parametric", []):
            model = band_result.get("model")
            pso = band_result.get("pso_params")
            if model and pso:
                if model not in params_by_model:
                    params_by_model[model] = []
                params_by_model[model].append(pso)

    # Compute mean and std for each model
    distributions = {}
    for model, all_params in params_by_model.items():
        arr = np.array(all_params)
        distributions[model] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "n_params": arr.shape[1],
            "n_samples": len(arr),
        }
    return distributions


def generate_observation_times(n_obs=50, baseline_days=100):
    """Generate ZTF-like observation cadence."""
    # Random cadence with ~3-day gaps, some clustering
    dt = np.random.exponential(3.0, n_obs)
    times = np.cumsum(dt)
    times = times[times <= baseline_days]
    if len(times) < 5:
        times = np.sort(np.random.uniform(0, baseline_days, max(5, n_obs // 2)))
    return times


def generate_one_source(model_name, params, times, rng):
    """Generate a synthetic multi-band light curve from a parametric model."""
    from lightcurve_fitting import eval_model

    n_obs = len(times)

    # Assign bands randomly (alternating with some randomness, like ZTF)
    bands = rng.choice([0, 1], size=n_obs, p=[0.45, 0.55])

    # Evaluate model at all times
    try:
        flux = np.array(eval_model(model_name, params.tolist(), times.tolist()))
    except Exception:
        return None

    # Skip if flux is all zero/nan/negative
    if not np.isfinite(flux).all() or flux.max() <= 0:
        return None

    # Convert to magnitude
    flux_positive = np.clip(flux, 1e-10, None)
    mag = -2.5 * np.log10(flux_positive)

    # Add realistic noise
    mag_err = np.maximum(ZTF_MAG_ERR_FLOOR, 0.1 * 10 ** (0.4 * (mag - 19)))
    mag_noisy = mag + rng.normal(0, mag_err)

    # Apply magnitude limit
    detectable = mag_noisy < ZTF_MAG_LIMIT
    if detectable.sum() < 3:
        return None

    times = times[detectable]
    mag_noisy = mag_noisy[detectable]
    mag_err = mag_err[detectable]
    bands = bands[detectable]

    # Build the feature tensor (same format as preprocess_alerts output)
    L = len(times)
    dt = times - times[0]
    dt_prev = np.zeros(L, dtype=np.float32)
    dt_prev[1:] = np.diff(times).astype(np.float32)

    logflux = (-0.4 * mag_noisy).astype(np.float32)
    logflux_err = (0.4 * mag_err).astype(np.float32)

    log_dt = np.log1p(dt).astype(np.float32)
    log_dt_prev = np.log1p(dt_prev).astype(np.float32)

    band_idx = bands.astype(np.int64)
    one_hot = np.eye(3, dtype=np.float32)[band_idx]

    base = np.column_stack([log_dt, log_dt_prev, logflux, logflux_err])

    # Metadata: zeros for synthetic (no real alert metadata)
    from dataset import N_META

    meta = np.zeros((L, N_META), dtype=np.float32)

    x = np.concatenate([base, one_hot, meta], axis=1).astype(np.float32)

    # No real image for synthetic sources
    image = np.zeros((3, 63, 63), dtype=np.float32)

    return x, image


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic light curves from parametric models"
    )
    parser.add_argument(
        "--fitting-dir",
        required=True,
        help="Path to fitting_results/ for parameter distributions",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--n-per-model",
        type=int,
        default=500,
        help="Number of synthetic sources per model",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Afterglow", "MetzgerKN", "Magnetar", "ShockCooling"],
        help="Models to generate (default: underrepresented classes)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    print("Loading parameter distributions from real fits...")
    distributions = load_param_distributions(args.fitting_dir)
    for model, dist in distributions.items():
        print(f"  {model}: {dist['n_samples']} samples, {dist['n_params']} params")

    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    for model_name in args.models:
        label = MODEL_TO_LABEL.get(model_name, 5)

        if model_name in distributions:
            dist = distributions[model_name]
            mean, std = dist["mean"], dist["std"]
        else:
            # No real fits for this model — use broader priors
            print(f"  {model_name}: no fitted distribution, using broad prior")
            # Use Bazin's param count as default
            n_params = distributions.get("Bazin", {"n_params": 6})["n_params"]
            mean = np.zeros(n_params)
            std = np.ones(n_params) * 2.0

        n_generated = 0
        n_attempts = 0
        max_attempts = args.n_per_model * 10

        while n_generated < args.n_per_model and n_attempts < max_attempts:
            n_attempts += 1

            # Sample parameters from the distribution
            params = mean + rng.randn(len(mean)) * std * 1.5  # slightly broader

            # Generate observation times
            times = generate_observation_times(
                n_obs=rng.randint(20, 100), baseline_days=rng.uniform(30, 200)
            )

            result = generate_one_source(model_name, params, times, rng)
            if result is None:
                continue

            x, image = result
            oid = f"SYN_{model_name}_{n_generated:05d}"
            out_path = os.path.join(args.output_dir, "train", f"{oid}.npz")
            np.savez_compressed(
                out_path,
                x=x,
                images=np.zeros((len(x), 3, 63, 63), dtype=np.float32),
                has_image=np.zeros(len(x), dtype=np.float32),
                label=np.int64(label),
            )
            n_generated += 1

        print(
            f"  {model_name}: generated {n_generated}/{args.n_per_model} "
            f"(label={label}, {n_attempts} attempts)"
        )

    total = len(list(Path(os.path.join(args.output_dir, "train")).glob("*.npz")))
    print(f"\nTotal synthetic sources: {total}")


if __name__ == "__main__":
    main()
