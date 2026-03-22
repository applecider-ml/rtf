"""
Generate realistic synthetic light curves using survey-sim with ZTF cadence.

Requires the survey-sim environment:
    module load gcc/13.2.0 openmpi/4.1.6 hdf5/1.14.3 python/3.11.5
    source /fred/oz480/mcoughli/envs/survey-sim/bin/activate

Produces NPZ files in the same format as preprocess_alerts.py output,
ready for use with --synthetic-dir in train.py.

Usage:
    python src/generate_surveysim.py \
        --ztf-dir /fred/oz480/mcoughli/simulations/ztf_boom \
        --output-dir data_synthetic \
        --populations kilonova afterglow tde snia snii \
        --n-per-pop 1000
"""

import argparse
import os

import numpy as np


# Label mapping: population type → coarse class
LABEL_MAP = {
    "Kilonova": 5,  # "other"
    "Afterglow": 5,  # "other"
    "SNIa": 0,  # SNIa
    "SNII": 1,  # SNcc
    "TDE": 4,  # TDE
}

# Model mapping: population name → (PopulationClass, model_name, model_key)
# Model key MUST match TransientType::Display from survey-sim Rust code:
#   Kilonova, SNIa, SNII, SNIbc, TDE, FBOT, Afterglow
POP_CONFIG = {
    "kilonova": {
        "pop_class": "KilonovaPopulation",
        "model_name": "MetzgerKN",
        "model_key": "Kilonova",
        "n_inject": 100000,
    },
    "afterglow_onaxis": {
        "pop_class": "OnAxisGrbPopulation",
        "model_name": None,  # uses FiestaAfterglowModel
        "model_key": "Afterglow",
        "n_inject": 100000,
        "grb_csv": True,
        "use_fiesta": True,
    },
    "afterglow_offaxis": {
        "pop_class": "OffAxisGrbPopulation",
        "model_name": None,
        "model_key": "Afterglow",
        "n_inject": 500000,
        "grb_csv": True,
        "use_fiesta": True,
        "rate": 800.0,
        "z_max": 1.0,
    },
    "tde": {
        "pop_class": "TdePopulation",
        "model_name": "Tde",
        "model_key": "TDE",
        "n_inject": 50000,
    },
    "snia": {
        "pop_class": "SupernovaIaPopulation",
        "model_name": "Bazin",
        "model_key": "SNIa",
        "n_inject": 10000,
    },
    "snii": {
        "pop_class": "SupernovaIIPopulation",
        "model_name": "Villar",
        "model_key": "SNII",
        "n_inject": 10000,
    },
}

# Metadata channels (zeros for synthetic — no real alert metadata)
N_META = 30


def photometry_to_npz(times, mags, mag_errs, bands, label):
    """Convert survey-sim photometry to RTF NPZ format."""
    L = len(times)
    if L < 2:
        return None

    # Sort by time
    order = np.argsort(times)
    times = np.array(times)[order]
    mags = np.array(mags)[order]
    mag_errs = np.array(mag_errs)[order]
    bands = [bands[i] for i in order]

    # Time features
    dt = (times - times[0]).astype(np.float32)
    dt_prev = np.zeros(L, dtype=np.float32)
    dt_prev[1:] = np.diff(times).astype(np.float32)

    log_dt = np.log1p(dt).astype(np.float32)
    log_dt_prev = np.log1p(dt_prev).astype(np.float32)

    # Flux features
    logflux = (-0.4 * mags).astype(np.float32)
    logflux_err = (0.4 * mag_errs).astype(np.float32)

    # Band encoding
    band_map = {"g": 0, "r": 1, "i": 2, "ztfg": 0, "ztfr": 1, "ztfi": 2}
    band_idx = np.array([band_map.get(b, 0) for b in bands], dtype=np.int64)
    one_hot = np.eye(3, dtype=np.float32)[band_idx]

    base = np.column_stack([log_dt, log_dt_prev, logflux, logflux_err])
    meta = np.zeros((L, N_META), dtype=np.float32)

    x = np.concatenate([base, one_hot, meta], axis=1).astype(np.float32)

    # No real images for synthetic sources
    images = np.zeros((L, 3, 63, 63), dtype=np.float32)
    has_image = np.zeros(L, dtype=np.float32)

    return {
        "x": x,
        "images": images,
        "has_image": has_image,
        "label": np.int64(label),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic light curves with survey-sim"
    )
    parser.add_argument(
        "--ztf-dir",
        required=True,
        help="Directory with ZTF HDF5 files (ztf_YYYYMM.h5)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--populations",
        nargs="+",
        default=["kilonova", "afterglow"],
        choices=list(POP_CONFIG.keys()),
    )
    parser.add_argument(
        "--n-per-pop",
        type=int,
        default=500,
        help="Target detections per population",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ztf-start", default="201803", help="ZTF survey start (YYYYMM)"
    )
    parser.add_argument("--ztf-end", default="202103", help="ZTF survey end (YYYYMM)")
    args = parser.parse_args()

    import survey_sim as ss

    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    print(f"Loading ZTF survey from {args.ztf_dir}...")
    survey = ss.load_ztf_survey_local(
        args.ztf_dir, start=args.ztf_start, end=args.ztf_end
    )
    print("Survey loaded.")

    criteria = ss.DetectionCriteria(min_detections=3, min_bands=1, snr_threshold=3.0)

    total_generated = 0

    for pop_name in args.populations:
        config = POP_CONFIG[pop_name]
        label = LABEL_MAP.get(config["model_key"], 5)

        # Create population
        pop_cls = getattr(ss, config["pop_class"])
        pop = pop_cls()

        # Create model
        model = ss.ParametricModel(model_name=config["model_name"])

        # Run simulation — inject enough to get n_per_pop detections
        n_inject = max(config["n_inject"], args.n_per_pop * 100)
        print(
            f"\n{pop_name}: injecting {n_inject} with {config['model_name']} model..."
        )

        pipeline = ss.SimulationPipeline(
            survey=survey,
            populations=[pop],
            models={config["model_key"]: model},
            detection=criteria,
            n_transients=n_inject,
            seed=args.seed,
        )
        result = pipeline.run()
        sources = result.sources()

        # Convert detections to NPZ
        n_saved = 0
        for s in sources:
            if n_saved >= args.n_per_pop:
                break
            t, m, e, b = s.photometry()
            npz_data = photometry_to_npz(t, m, e, b, label)
            if npz_data is None:
                continue

            oid = f"SYN_{pop_name}_{n_saved:05d}"
            out_path = os.path.join(args.output_dir, "train", f"{oid}.npz")
            np.savez_compressed(out_path, **npz_data)
            n_saved += 1

        print(
            f"  {pop_name}: {n_saved} saved from {len(sources)} detected "
            f"({result.n_simulated} injected, label={label})"
        )
        total_generated += n_saved

    print(f"\nTotal synthetic sources: {total_generated}")


if __name__ == "__main__":
    main()
