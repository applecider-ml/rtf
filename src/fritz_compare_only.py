#!/usr/bin/env python3
"""Query Fritz for ALL scored objects from the existing tarball predictions CSV.

Skips the tarball extraction + TEMPO inference entirely — just reads the
already-saved CSV and queries Fritz for every objectId.

Usage:
    python fritz_compare_only.py \
        --predictions /work/hdd/bcrv/kmajithia/tempo/tarball_comparison/tempo_tarball_predictions.csv \
        --fritz-token YOUR_TOKEN \
        --output-dir /work/hdd/bcrv/kmajithia/tempo/tarball_comparison
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

CLASS_NAMES = ["SNI", "SNII", "TDE", "AGN", "CV"]

# Fritz class name -> TEMPO broad class
FRITZ_TO_TEMPO = {
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
    "SLSN-I": "SNI",
    "SN II": "SNII",
    "SN IIP": "SNII",
    "SN IIL": "SNII",
    "SN IIn": "SNII",
    "SN IIb": "SNII",
    "SN II-pec": "SNII",
    "SN IIn-pec": "SNII",
    "SLSN-II": "SNII",
    "TDE": "TDE",
    "Tidal Disruption Event": "TDE",
    "AGN": "AGN",
    "QSO": "AGN",
    "Blazar": "AGN",
    "Seyfert": "AGN",
    "LINER": "AGN",
    "CV": "CV",
    "Cataclysmic Variable": "CV",
    "Nova": "CV",
    "Cataclysmic": "CV",
    "CV*": "CV",
    "dwarf nova": "CV",
}


def query_fritz_classifications(object_ids, token):
    """Query Fritz API for classifications of given objectIds."""
    base = "https://fritz.science"
    headers = {"Authorization": f"token {token}"}
    results = {}
    n_404 = 0
    n_err = 0

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
                    best = max(data, key=lambda c: c.get("probability", 0))
                    results[oid] = {
                        "fritz_class": best.get("classification", "Unknown"),
                        "fritz_prob": best.get("probability", 0),
                        "fritz_ml": best.get("ml", False),
                    }
            elif r.status_code == 404:
                n_404 += 1
            else:
                n_err += 1

            if (i + 1) % 200 == 0:
                print(
                    f"  [{i + 1}/{len(object_ids)}] found={len(results)}, 404={n_404}, err={n_err}",
                    flush=True,
                )

            time.sleep(0.2)  # rate limit

        except Exception:
            n_err += 1
            time.sleep(1)

    print(
        f"  DONE: {len(object_ids)} queried, "
        f"{len(results)} classified, {n_404} not found, {n_err} errors"
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="/work/hdd/bcrv/kmajithia/tempo/tarball_comparison/tempo_tarball_predictions.csv",
    )
    parser.add_argument("--fritz-token", default=os.environ.get("FRITZ_TOKEN", ""))
    parser.add_argument("--output-dir", default="/work/hdd/bcrv/kmajithia/tempo/tarball_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.fritz_token:
        print("ERROR: --fritz-token or FRITZ_TOKEN env var required")
        return

    # Load existing predictions
    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df)} predictions from {args.predictions}")
    print("Class distribution:")
    for cls in CLASS_NAMES:
        n = (df["tempo_class"] == cls).sum()
        print(f"  {cls}: {n}")

    # Query Fritz for ALL objects
    all_oids = df["objectId"].tolist()
    print(f"\nQuerying Fritz for ALL {len(all_oids)} objects...")
    fritz_results = query_fritz_classifications(all_oids, args.fritz_token)
    print(f"\nFritz returned classifications for {len(fritz_results)} objects")

    if not fritz_results:
        print("No Fritz classifications found. Exiting.")
        return

    # Build comparison
    comp_rows = []
    for oid, fritz_info in fritz_results.items():
        row = df[df["objectId"] == oid].iloc[0].to_dict()
        fritz_class_raw = fritz_info["fritz_class"]
        fritz_broad = FRITZ_TO_TEMPO.get(fritz_class_raw, None)
        row["fritz_class_raw"] = fritz_class_raw
        row["fritz_class_broad"] = fritz_broad
        row["fritz_prob"] = fritz_info["fritz_prob"]
        row["fritz_ml"] = fritz_info["fritz_ml"]
        row["match"] = (fritz_broad == row["tempo_class"]) if fritz_broad else None
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)
    comp_csv = out_dir / "tempo_vs_fritz_full.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"Saved: {comp_csv}")

    # Filter to mappable
    mapped = comp_df[comp_df["fritz_class_broad"].notna()].copy()
    unmapped = comp_df[comp_df["fritz_class_broad"].isna()]

    print(f"\n{'=' * 60}")
    print("TEMPO vs Fritz — Full Comparison")
    print(f"{'=' * 60}")
    print(f"Fritz classified:      {len(fritz_results)}")
    print(f"Mappable to TEMPO:     {len(mapped)}")
    if len(unmapped) > 0:
        print(f"Unmappable classes:    {len(unmapped)}")
        print(f"  Classes: {unmapped['fritz_class_raw'].value_counts().to_dict()}")

    if len(mapped) == 0:
        print("No mappable objects. Exiting.")
        return

    n_match = mapped["match"].sum()
    agreement = n_match / len(mapped)
    print(f"Agreement:             {n_match}/{len(mapped)} ({100 * agreement:.1f}%)")

    # Per-class
    print("\nPer Fritz class:")
    for cls in CLASS_NAMES:
        cls_mask = mapped["fritz_class_broad"] == cls
        if cls_mask.sum() > 0:
            cls_match = mapped.loc[cls_mask, "match"].sum()
            cls_total = cls_mask.sum()
            print(f"  {cls}: {cls_match}/{cls_total} ({100 * cls_match / cls_total:.1f}%)")

    # Uncertainty analysis: does thresholding help?
    print("\nUncertainty thresholding analysis:")
    for thresh in [0.20, 0.25, 0.30]:
        confident = mapped[mapped["tempo_uncertainty"] <= thresh]
        if len(confident) > 0:
            acc = confident["match"].sum() / len(confident)
            coverage = len(confident) / len(mapped)
            print(
                f"  threshold={thresh:.2f}: "
                f"accuracy={100 * acc:.1f}%, "
                f"coverage={100 * coverage:.1f}% "
                f"({len(confident)}/{len(mapped)} objects)"
            )

    # Disagreements
    disagree = mapped[mapped["match"] == False]
    if len(disagree) > 0:
        print(f"\nDisagreements ({len(disagree)}):")
        for _, row in disagree.head(30).iterrows():
            print(
                f"  {row['objectId']}: "
                f"TEMPO={row['tempo_class']} ({row['tempo_max_prob']:.2f}) "
                f"vs Fritz={row['fritz_class_raw']} "
                f"| unc={row['tempo_uncertainty']:.3f}"
            )

    # Confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = mapped["fritz_class_broad"].values
    y_pred = mapped["tempo_class"].values

    present = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=present)

    print(f"\n{classification_report(y_true, y_pred, labels=present, zero_division=0)}")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("TEMPO vs Fritz — Confusion Matrix (Full Tarball)", fontsize=14, fontweight="bold")
    ax.set_xlabel("TEMPO Predicted")
    ax.set_ylabel("Fritz Classification")
    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present)
    ax.set_yticklabels(present)
    fig.colorbar(im)
    for i in range(len(present)):
        for j in range(len(present)):
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
    fig.savefig(out_dir / "tempo_vs_fritz_confusion_full.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'tempo_vs_fritz_confusion_full.png'}")

    # Summary JSON
    summary = {
        "predictions_file": str(args.predictions),
        "total_scored": len(df),
        "fritz_queried": len(all_oids),
        "fritz_found": len(fritz_results),
        "fritz_mappable": len(mapped),
        "fritz_unmappable": len(unmapped),
        "agreement": round(float(agreement), 4),
        "thresholding": {
            f"thresh_{t}": {
                "accuracy": round(
                    float(
                        mapped[mapped["tempo_uncertainty"] <= t]["match"].sum()
                        / max(len(mapped[mapped["tempo_uncertainty"] <= t]), 1)
                    ),
                    4,
                ),
                "coverage": round(
                    float(len(mapped[mapped["tempo_uncertainty"] <= t]) / len(mapped)), 4
                ),
            }
            for t in [0.20, 0.25, 0.30]
        },
    }
    with open(out_dir / "comparison_summary_full.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_dir / 'comparison_summary_full.json'}")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
