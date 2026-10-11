#!/usr/bin/env python3
"""Run TEMPO on a ZTF alert tarball and compare against Fritz classifications.



Usage:
    python score_tarball_fritz.py \
        --tarball /path/to/ztf_public_YYYYMMDD.tar.gz \
        --onnx /path/to/tempo_classifier.onnx \
        --fritz-token YOUR_TOKEN \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import requests

TEMPO_ROOT = os.environ.get(
    "TEMPO_ROOT",
    "/projects/bcrv/ffontinelenunes/transformer/tempo",
)
if TEMPO_ROOT not in sys.path:
    sys.path.insert(0, TEMPO_ROOT)

try:
    import fastavro
except ImportError:
    print("Installing fastavro...")
    os.system(f"{sys.executable} -m pip install fastavro -q")
    import fastavro

# ---------------------------------------------------------------------------
# TEMPO config
# ---------------------------------------------------------------------------
MAX_LEN = 257
NUM_CLASSES = 5
CLASS_NAMES = ["SNI", "SNII", "TDE", "AGN", "CV"]
FEAT_MEAN = np.array([3.2246506, 0.75406283, 1.8746188, 0.05986891], dtype=np.float32)
FEAT_STD = np.array([1.1197281, 0.72683305, 0.4150701, 0.03053664], dtype=np.float32)

# Fritz class name -> TEMPO broad class
FRITZ_TO_TEMPO = {
    # SNI (Type I)
    "SN Ia": "SNI",
    "SN Ia-91T-like": "SNI",
    "SN Ia-91bg-like": "SNI",
    "SN Ia-CSM": "SNI",
    "SN Ia-pec": "SNI",
    "SN Ia-02cx-like": "SNI",
    "SN Iax[02cx-like]": "SNI",
    "SN Ia-SC": "SNI",
    "SN Ib": "SNI",
    "SN Ib-pec": "SNI",
    "SN Ib/c": "SNI",
    "SN Ic": "SNI",
    "SN Ic-BL": "SNI",
    "SN Ic-pec": "SNI",
    "SN I": "SNI",
    # SNII (Type II)
    "SN II": "SNII",
    "SN IIP": "SNII",
    "SN IIL": "SNII",
    "SN IIn": "SNII",
    "SN IIb": "SNII",
    "SN II-pec": "SNII",
    "SN IIn-pec": "SNII",
    "SLSN-II": "SNII",
    "SLSN-I": "SNI",
    # TDE
    "TDE": "TDE",
    "Tidal Disruption Event": "TDE",
    # AGN
    "AGN": "AGN",
    "QSO": "AGN",
    "Blazar": "AGN",
    "Seyfert": "AGN",
    "LINER": "AGN",
    # CV
    "CV": "CV",
    "Cataclysmic Variable": "CV",
    "Nova": "CV",
    "Cataclysmic": "CV",
    "CV*": "CV",
    "dwarf nova": "CV",
}

# fid -> band_id (ZTF filter IDs)
FID_TO_BAND = {1: 0, 2: 1, 3: 2}  # g=0, r=1, i=2


def extract_photometry_from_alert(alert):
    """Extract (jd, mag, sigmag, band_id) tuples from a single avro alert."""
    detections = []
    cand = alert.get("candidate", {})

    # Current candidate
    if cand.get("magpsf") is not None and cand.get("sigmapsf") is not None:
        fid = cand.get("fid", 2)
        bid = FID_TO_BAND.get(fid, 1)
        detections.append((cand["jd"], float(cand["magpsf"]), float(cand["sigmapsf"]), bid))

    # Previous candidates
    for prv in alert.get("prv_candidates") or []:
        if prv.get("magpsf") is not None and prv.get("sigmapsf") is not None:
            fid = prv.get("fid", 2)
            bid = FID_TO_BAND.get(fid, 1)
            detections.append((prv["jd"], float(prv["magpsf"]), float(prv["sigmapsf"]), bid))

    return detections


def build_tempo_input(detections):
    """Build TEMPO input tensors from raw photometry detections.

    Args:
        detections: list of (jd, mag, sigmag, band_id) tuples

    Returns:
        x: (1, 257, 5) normalized
        pad_mask: (1, 257) bool
        global_features: (1, 24) float
    """
    # Drop i-band
    dets = [(jd, mag, sig, b) for jd, mag, sig, b in detections if b != 2]

    if len(dets) < 3:
        return None, None, None

    # Sort by JD
    dets.sort(key=lambda x: x[0])

    # Truncate to MAX_LEN
    if len(dets) > MAX_LEN:
        dets = dets[-MAX_LEN:]

    n_obs = len(dets)
    jd0 = dets[0][0]

    # Build raw features: [log1p(dt), log1p(dt_prev), log_flux, log_flux_err, band_id]
    raw = np.zeros((n_obs, 5), dtype=np.float32)
    for i, (jd, mag, sig, bid) in enumerate(dets):
        dt = jd - jd0
        dt_prev = jd - dets[i - 1][0] if i > 0 else 0.0

        raw[i, 0] = np.log1p(dt)  # log1p(dt_first)
        raw[i, 1] = np.log1p(dt_prev)  # log1p(dt_prev)
        raw[i, 2] = (23.9 - mag) / 2.5  # log10(flux_uJy)
        raw[i, 3] = 0.4 * sig  # log_flux_err
        raw[i, 4] = bid  # band_id

    # Compute global features from raw
    gf = compute_global_features(raw)

    # Normalize continuous channels
    x_norm = raw.copy()
    x_norm[:, :4] = (raw[:, :4] - FEAT_MEAN) / (FEAT_STD + 1e-8)

    # Pad to MAX_LEN
    x_padded = np.zeros((1, MAX_LEN, 5), dtype=np.float32)
    x_padded[0, :n_obs] = x_norm

    pad_mask = np.ones((1, MAX_LEN), dtype=bool)
    pad_mask[0, :n_obs] = False

    g = gf.reshape(1, 24)

    return x_padded, pad_mask, g


def compute_global_features(raw):
    """Compute 24 physics features from raw (unnormalized) token array."""
    out = np.zeros(24, dtype=np.float32)
    if len(raw) == 0:
        return out

    n_obs = len(raw)
    dt_first = np.expm1(raw[:, 0]).clip(min=0)
    dt_prev = np.expm1(raw[:, 1]).clip(min=0)
    logf = raw[:, 2]
    bands = raw[:, 4].astype(int)

    duration = dt_first.max()
    counts = np.array([np.sum(bands == b) for b in range(3)], dtype=np.float32)
    amplitude = logf.max() - logf.min()

    # Per-band mean flux
    band_means = np.zeros(3)
    for b in range(3):
        mask = bands == b
        if mask.sum() > 0:
            band_means[b] = logf[mask].mean()

    # Basic (8)
    out[0] = duration
    out[1] = float(n_obs)
    out[2:5] = counts
    out[5] = amplitude
    out[6] = band_means[0] - band_means[1]  # color_gr
    out[7] = band_means[1] - band_means[2]  # color_ri

    # Enhanced (8)
    idx_peak = int(np.argmax(logf))
    peak_t = dt_first[idx_peak]
    out[8] = peak_t / max(duration, 1e-6)  # peak_frac
    out[9] = logf[idx_peak]  # peak_flux
    out[10] = float(np.median(dt_prev))  # med_dt_prev
    out[11] = float(np.std(logf))  # std_flux
    out[12] = float(np.percentile(logf, 90) - np.percentile(logf, 10))  # iqr

    # Slopes
    def safe_slope(x, y):
        if len(x) < 2:
            return 0.0
        xm, ym = x.mean(), y.mean()
        sxx = ((x - xm) ** 2).sum()
        if sxx < 1e-12:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / sxx)

    out[13] = safe_slope(dt_first[: idx_peak + 1], logf[: idx_peak + 1])
    out[14] = safe_slope(dt_first[idx_peak:], logf[idx_peak:])
    out[15] = out[13] / max(abs(out[14]), 1e-6)

    # Physics (8)
    n_safe = max(n_obs, 1)
    out[16] = counts[0] / n_safe
    out[17] = counts[1] / n_safe
    out[18] = counts[2] / n_safe

    for b in range(3):
        mask = bands == b
        if mask.sum() >= 2:
            out[19 + b] = safe_slope(dt_first[mask], logf[mask])
    out[22] = out[19] - out[20]  # color_gr_slope
    out[23] = out[20] - out[21]  # color_ri_slope

    return out


def query_fritz_classifications(object_ids, token, batch_size=50):
    """Query Fritz API for classifications of given objectIds."""
    base = "https://fritz.science"
    headers = {"Authorization": f"token {token}"}
    results = {}

    for i, oid in enumerate(object_ids):
        try:
            r = requests.get(
                f"{base}/api/sources/{oid}/classifications",
                headers=headers,
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    # Take highest probability classification
                    best = max(data, key=lambda c: c.get("probability", 0))
                    results[oid] = {
                        "fritz_class": best.get("classification", "Unknown"),
                        "fritz_prob": best.get("probability", 0),
                        "fritz_ml": best.get("ml", False),
                    }
            elif r.status_code == 404:
                pass  # Not a Fritz source
            else:
                pass

            if (i + 1) % 100 == 0:
                print(f"  Fritz query: {i + 1}/{len(object_ids)} ({len(results)} found)")

            time.sleep(0.3)  # Rate limiting

        except Exception:
            time.sleep(1)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tarball", help="Path to ztf_public_YYYYMMDD.tar.gz")
    parser.add_argument("--avro-dir", help="Path to extracted avro directory")
    parser.add_argument(
        "--onnx", default="/work/hdd/bcrv/kmajithia/tempo/onnx/tempo_classifier.onnx"
    )
    parser.add_argument("--fritz-token", default=os.environ.get("FRITZ_TOKEN", ""))
    parser.add_argument(
        "--fritz-sample", type=int, default=500, help="Number of objects to query Fritz for"
    )
    parser.add_argument("--max-alerts", type=int, default=50000)
    parser.add_argument("--output-dir", default="/work/hdd/bcrv/kmajithia/tempo/tarball_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1+2. Read alerts directly (stream from tarball OR read avro dir)
    # ------------------------------------------------------------------
    object_detections = defaultdict(list)
    n_read = 0

    if args.avro_dir:
        avro_dir = Path(args.avro_dir)
        avro_files = sorted(avro_dir.glob("*.avro"))
        print(f"Reading {len(avro_files)} avro files from directory...")
        for af in avro_files[: args.max_alerts]:
            try:
                with open(af, "rb") as f:
                    for alert in fastavro.reader(f):
                        oid = alert.get("objectId", "")
                        if not oid:
                            continue
                        dets = extract_photometry_from_alert(alert)
                        object_detections[oid].extend(dets)
                        n_read += 1
            except Exception:
                continue
            if n_read % 10000 == 0 and n_read > 0:
                print(f"  Read {n_read} alerts, {len(object_detections)} unique objects")

    elif args.tarball:
        # Stream avro directly from tarball in memory (no disk extraction)
        print(f"Streaming from tarball: {args.tarball}")
        import io

        with tarfile.open(args.tarball, "r:gz") as tar:
            for member in tar:
                if not member.name.endswith(".avro") or not member.isfile():
                    continue
                if n_read >= args.max_alerts:
                    break
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    avro_bytes = io.BytesIO(f.read())
                    for alert in fastavro.reader(avro_bytes):
                        oid = alert.get("objectId", "")
                        if not oid:
                            continue
                        dets = extract_photometry_from_alert(alert)
                        object_detections[oid].extend(dets)
                        n_read += 1
                except Exception:
                    continue

                if n_read % 5000 == 0 and n_read > 0:
                    print(
                        f"  Streamed {n_read} alerts, {len(object_detections)} unique objects",
                        flush=True,
                    )
    else:
        print("ERROR: Provide --tarball or --avro-dir")
        return

    print(f"Total: {n_read} alerts, {len(object_detections)} unique objects")

    # Deduplicate detections per object (same JD = same detection)
    for oid in object_detections:
        seen = set()
        unique = []
        for det in object_detections[oid]:
            key = (round(det[0], 6), det[3])  # (jd, band)
            if key not in seen:
                seen.add(key)
                unique.append(det)
        object_detections[oid] = unique

    # ------------------------------------------------------------------
    # 3. Run TEMPO on all objects
    # ------------------------------------------------------------------
    print(f"\nLoading ONNX: {args.onnx}")
    sess = ort.InferenceSession(args.onnx)
    input_names = [inp.name for inp in sess.get_inputs()]

    print("Running TEMPO inference...")
    predictions = []
    n_skipped = 0

    for i, (oid, dets) in enumerate(object_detections.items()):
        x, mask, g = build_tempo_input(dets)
        if x is None:
            n_skipped += 1
            continue

        feed = {input_names[0]: x, input_names[1]: mask, input_names[2]: g}
        evidence = sess.run(None, feed)[0][0]
        alpha = evidence + 1.0
        S = alpha.sum()
        probs = alpha / S
        pred_class = int(np.argmax(probs))
        uncertainty = NUM_CLASSES / S

        predictions.append(
            {
                "objectId": oid,
                "n_detections": len(dets),
                "tempo_class": CLASS_NAMES[pred_class],
                "tempo_class_idx": pred_class,
                "tempo_max_prob": float(probs.max()),
                "tempo_uncertainty": float(uncertainty),
                **{f"prob_{CLASS_NAMES[j]}": float(probs[j]) for j in range(NUM_CLASSES)},
            }
        )

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(object_detections)}")

    print(f"Scored: {len(predictions)}, Skipped (too few obs): {n_skipped}")

    df = pd.DataFrame(predictions)
    csv_path = out_dir / "tempo_tarball_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print TEMPO class distribution
    print("\nTEMPO class distribution:")
    for cls in CLASS_NAMES:
        n = (df["tempo_class"] == cls).sum()
        print(f"  {cls}: {n} ({100 * n / len(df):.1f}%)")

    # ------------------------------------------------------------------
    # 4. Query Fritz for classifications
    # ------------------------------------------------------------------
    if not args.fritz_token:
        print("\n[SKIP] No Fritz token provided. Set --fritz-token or FRITZ_TOKEN env var.")
        print("Skipping Fritz comparison.")
    else:
        print(f"\nQuerying Fritz for {args.fritz_sample} objects...")
        sample_oids = (
            df["objectId"].sample(min(args.fritz_sample, len(df)), random_state=42).tolist()
        )

        fritz_results = query_fritz_classifications(sample_oids, args.fritz_token)
        print(f"Fritz returned classifications for {len(fritz_results)} objects")

        if fritz_results:
            # Build comparison DataFrame
            comp_rows = []
            for oid, fritz_info in fritz_results.items():
                row = df[df["objectId"] == oid].iloc[0].to_dict()
                fritz_class_raw = fritz_info["fritz_class"]
                fritz_broad = FRITZ_TO_TEMPO.get(fritz_class_raw, None)
                row["fritz_class_raw"] = fritz_class_raw
                row["fritz_class_broad"] = fritz_broad
                row["fritz_prob"] = fritz_info["fritz_prob"]
                row["match"] = (fritz_broad == row["tempo_class"]) if fritz_broad else None
                comp_rows.append(row)

            comp_df = pd.DataFrame(comp_rows)
            comp_csv = out_dir / "tempo_vs_fritz.csv"
            comp_df.to_csv(comp_csv, index=False)
            print(f"Saved: {comp_csv}")

            # Filter to objects where we can map Fritz class
            mapped = comp_df[comp_df["fritz_class_broad"].notna()]
            if len(mapped) > 0:
                n_match = mapped["match"].sum()
                agreement = n_match / len(mapped)

                print(f"\n{'=' * 60}")
                print("TEMPO vs Fritz Comparison")
                print(f"{'=' * 60}")
                print(f"Objects with Fritz classification: {len(fritz_results)}")
                print(f"Mappable to TEMPO classes: {len(mapped)}")
                print(f"Agreement: {n_match}/{len(mapped)} ({100 * agreement:.1f}%)")

                # Per-class breakdown
                print("\nPer-class agreement:")
                for cls in CLASS_NAMES:
                    cls_mask = mapped["fritz_class_broad"] == cls
                    if cls_mask.sum() > 0:
                        cls_match = mapped.loc[cls_mask, "match"].sum()
                        cls_total = cls_mask.sum()
                        print(
                            f"  {cls}: {cls_match}/{cls_total} ({100 * cls_match / cls_total:.1f}%)"
                        )

                # Disagreements
                disagree = mapped[mapped["match"] == False]
                if len(disagree) > 0:
                    print(f"\nDisagreements ({len(disagree)}):")
                    for _, row in disagree.head(20).iterrows():
                        print(
                            f"  {row['objectId']}: "
                            f"TEMPO={row['tempo_class']} ({row['tempo_max_prob']:.2f}) "
                            f"vs Fritz={row['fritz_class_raw']} "
                            f"| uncertainty={row['tempo_uncertainty']:.3f}"
                        )

                # Confusion matrix
                from sklearn.metrics import confusion_matrix

                y_true_labels = mapped["fritz_class_broad"].values
                y_pred_labels = mapped["tempo_class"].values

                present_classes = sorted(set(y_true_labels) | set(y_pred_labels))
                cm = confusion_matrix(y_true_labels, y_pred_labels, labels=present_classes)

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_title("TEMPO vs Fritz — Confusion Matrix", fontsize=14, fontweight="bold")
                ax.set_xlabel("TEMPO Predicted")
                ax.set_ylabel("Fritz Classification")
                ax.set_xticks(range(len(present_classes)))
                ax.set_yticks(range(len(present_classes)))
                ax.set_xticklabels(present_classes)
                ax.set_yticklabels(present_classes)
                fig.colorbar(im)
                for i in range(len(present_classes)):
                    for j in range(len(present_classes)):
                        color = "white" if cm[i, j] > cm.max() / 2 else "black"
                        ax.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center",
                            fontsize=12,
                            color=color,
                            fontweight="bold",
                        )
                fig.tight_layout()
                fig.savefig(out_dir / "tempo_vs_fritz_confusion.png", dpi=150)
                plt.close()
                print(f"\nSaved: {out_dir / 'tempo_vs_fritz_confusion.png'}")

                # Save summary
                summary = {
                    "tarball": args.tarball or args.avro_dir,
                    "total_alerts": n_read,
                    "unique_objects": len(object_detections),
                    "tempo_scored": len(predictions),
                    "fritz_queried": len(sample_oids),
                    "fritz_found": len(fritz_results),
                    "fritz_mappable": len(mapped),
                    "agreement": round(float(agreement), 4),
                }
                with open(out_dir / "comparison_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
