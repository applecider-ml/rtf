"""
Fetch sources from Fritz/SkyPortal groups and preprocess into RTF model format.

Queries the Fritz API for all sources in specified groups (SLSNe, Fast Transients,
LIONS), downloads their photometry, and converts to the exact (L, 37) tensor format
the RTF autoencoder expects.

Usage:
    python fetch_fritz_sources.py \
        --token YOUR_FRITZ_API_TOKEN \
        --group-ids 53,85,1876 \
        --output-dir ../fritz_sources/
"""

import argparse
import csv
import os
import time

import numpy as np
import requests

# ── The 30 alert metadata keys the RTF model uses (from dataset.py) ──
ALERT_META_KEYS = [
    "diffmaglim",
    "ra",
    "dec",
    "magpsf",
    "sigmapsf",
    "chipsf",
    "fwhm",
    "sky",
    "mindtoedge",
    "seeratio",
    "aimage",
    "bimage",
    "elong",
    "nneg",
    "nbad",
    "scorr",
    "dsnrms",
    "ssnrms",
    "magzpsci",
    "magzpsciunc",
    "magzpscirms",
    "nmatches",
    "clrcoeff",
    "clrcounc",
    "zpclrcov",
    "zpmed",
    "clrmed",
    "srmag1",
    "sgmag1",
    "simag1",
]

# Fritz band name → ZTF fid
BAND_TO_FID = {"ztfg": 1, "ztfr": 2, "ztfi": 3}

FRITZ_BASE = "https://fritz.science"

# Group ID → human-readable name
GROUP_NAMES = {
    53: "SLSNe",
    85: "Fast_Transients",
    1876: "LIONS",
}


def fritz_get(endpoint, token, params=None):
    """Make a GET request to the Fritz API with retry logic."""
    headers = {"Authorization": f"token {token}"}
    url = f"{FRITZ_BASE}{endpoint}"
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "success":
                return data["data"]
            else:
                print(f"    API error: {data.get('message', 'unknown')}")
                return None
        except Exception as e:
            print(f"    Request failed (attempt {attempt + 1}): {e}")
            time.sleep(2**attempt)
    return None


def fetch_source_classification(token, obj_id):
    """Fetch the best classification for a single source via dedicated endpoint."""
    data = fritz_get(f"/api/sources/{obj_id}/classifications", token)
    if data is None or not data:
        return None
    # Pick highest-probability classification
    best = max(data, key=lambda c: c.get("probability", 0))
    return {
        "classification": best.get("classification", ""),
        "probability": best.get("probability", 0),
        "classifier": "ML" if best.get("ml", False) else "human",
    }


def fetch_group_sources(token, group_id):
    """Fetch all source IDs from a Fritz group with pagination."""
    all_sources = []
    page = 1
    per_page = 500

    print(f"\n  Fetching sources from group {group_id} ({GROUP_NAMES.get(group_id, '?')})...")

    while True:
        data = fritz_get(
            "/api/sources",
            token,
            params={
                "group_ids": group_id,
                "numPerPage": per_page,
                "pageNumber": page,
                "includePhotometry": False,
                "includeSpectra": False,
            },
        )

        if data is None:
            print(f"    Failed to fetch page {page}")
            break

        sources = data.get("sources", [])
        if not sources:
            break

        for s in sources:
            all_sources.append(
                {
                    "id": s["id"],
                    "ra": s.get("ra"),
                    "dec": s.get("dec"),
                    "redshift": s.get("redshift"),
                    "classifications": [
                        c.get("classification", "") for c in s.get("classifications", [])
                    ],
                }
            )

        total = data.get("totalMatches", "?")
        print(f"    Page {page}: {len(sources)} sources (total: {total})")

        if len(sources) < per_page:
            break
        page += 1
        time.sleep(0.5)

    print(f"    Total: {len(all_sources)} sources from group {group_id}")
    return all_sources


def fetch_photometry(token, obj_id):
    """Fetch photometry for a single source."""
    data = fritz_get(f"/api/sources/{obj_id}/photometry", token)
    if data is None:
        return None
    return data


def photometry_to_tensor(phot_data):
    """Convert Fritz photometry JSON to the (L, 37) tensor format.

    The RTF model expects:
        Channels 0-3: log1p(dt), log1p(dt_prev), logflux, logflux_err
        Channels 4-6: one-hot band (g, r, i)
        Channels 7-36: 30 alert metadata keys (mostly zeros for Fritz data)
    """
    # Filter to ZTF photometry with valid mag
    points = []
    for p in phot_data:
        p.get("instrument_name", "")
        filt = p.get("filter", "")
        mag = p.get("mag")
        magerr = p.get("magerr")
        mjd = p.get("mjd")

        # Skip non-ZTF or invalid points
        if filt not in BAND_TO_FID:
            continue
        if mag is None or mjd is None:
            continue
        if magerr is None:
            magerr = 0.1  # default uncertainty

        points.append(
            {
                "mjd": float(mjd),
                "mag": float(mag),
                "magerr": float(magerr),
                "fid": BAND_TO_FID[filt],
                "ra": p.get("ra", 0.0),
                "dec": p.get("dec", 0.0),
            }
        )

    if len(points) < 3:
        return None

    # Sort by time
    points.sort(key=lambda p: p["mjd"])

    L = len(points)
    mjds = np.array([p["mjd"] for p in points], dtype=np.float64)
    jds = mjds + 2400000.5  # MJD → JD
    mags = np.array([p["mag"] for p in points], dtype=np.float32)
    magerrs = np.array([p["magerr"] for p in points], dtype=np.float32)
    fids = np.array([p["fid"] for p in points], dtype=np.int64)

    # Time features
    dt = (jds - jds[0]).astype(np.float32)
    dt_prev = np.zeros(L, dtype=np.float32)
    dt_prev[1:] = np.diff(jds).astype(np.float32)

    log_dt = np.log1p(dt)
    log_dt_prev = np.log1p(dt_prev)

    # Photometry features
    logflux = (-0.4 * mags).astype(np.float32)
    logflux_err = (0.4 * magerrs).astype(np.float32)

    # One-hot band (fid: 1=g, 2=r, 3=i → index 0, 1, 2)
    band_idx = (fids - 1).clip(0, 2).astype(np.int64)
    one_hot = np.eye(3, dtype=np.float32)[band_idx]

    # Base: (L, 4)
    base = np.column_stack([log_dt, log_dt_prev, logflux, logflux_err])

    # Metadata: (L, 30) — fill with available fields, rest is 0
    meta = np.zeros((L, len(ALERT_META_KEYS)), dtype=np.float32)
    for i, p in enumerate(points):
        # Fill ra, dec, magpsf, sigmapsf from photometry where available
        cand = {
            "ra": p.get("ra", 0.0),
            "dec": p.get("dec", 0.0),
            "magpsf": p["mag"],
            "sigmapsf": p["magerr"],
        }
        for j, key in enumerate(ALERT_META_KEYS):
            val = cand.get(key, 0.0)
            if val is None:
                val = 0.0
            try:
                meta[i, j] = float(val)
            except (ValueError, TypeError):
                meta[i, j] = 0.0

    # Concatenate: (L, 37)
    x = np.concatenate([base, one_hot, meta], axis=1).astype(np.float32)
    return x


def main():
    parser = argparse.ArgumentParser(description="Fetch Fritz sources and preprocess for RTF")
    parser.add_argument("--token", required=True, help="Fritz API token")
    parser.add_argument(
        "--group-ids", default="53,85,1876", help="Comma-separated group IDs (e.g. 53,85,1876)"
    )
    parser.add_argument("--output-dir", default="fritz_sources", help="Output directory")
    parser.add_argument(
        "--saved-objects",
        default=None,
        help="Path to existing anomaly_scores.csv — will also fetch classifications for these obj IDs",
    )
    parser.add_argument(
        "--max-sources", type=int, default=None, help="Max sources per group (for testing)"
    )
    args = parser.parse_args()

    group_ids = [int(g.strip()) for g in args.group_ids.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Fritz Source Fetcher + Preprocessor ===")
    print(f"Groups: {group_ids}")
    print(f"Output: {args.output_dir}")

    # Manifest CSV
    manifest_path = os.path.join(args.output_dir, "fritz_manifest.csv")
    manifest_rows = []

    total_fetched = 0
    total_preprocessed = 0

    for group_id in group_ids:
        group_name = GROUP_NAMES.get(group_id, f"group_{group_id}")
        sources = fetch_group_sources(args.token, group_id)
        if not sources:
            print(f"  No sources found for group {group_id}, skipping")
            continue

        print(f"\n  Processing {len(sources)} sources from {group_name}...")

        for idx, src in enumerate(sources):
            obj_id = src["id"]
            total_fetched += 1

            if args.max_sources and idx >= args.max_sources:
                print(f"    Reached --max-sources={args.max_sources}, stopping group")
                break

            # Resume support: skip already-downloaded sources
            out_path = os.path.join(args.output_dir, f"{obj_id}.npz")
            if os.path.exists(out_path):
                total_preprocessed += 1
                if (idx + 1) % 100 == 0:
                    print(f"    [{idx + 1}/{len(sources)}] {obj_id}: already exists, skipping")
                continue

            # Fetch photometry
            phot = fetch_photometry(args.token, obj_id)
            if phot is None:
                print(f"    [{idx + 1}/{len(sources)}] {obj_id}: no photometry, skipping")
                continue
            time.sleep(0.3)  # rate limit

            # Convert to tensor
            x = photometry_to_tensor(phot)
            if x is None:
                print(f"    [{idx + 1}/{len(sources)}] {obj_id}: <3 valid ZTF points, skipping")
                continue

            # Fetch richer classification via dedicated endpoint
            cls_info = fetch_source_classification(args.token, obj_id)
            time.sleep(0.2)  # rate limit

            # Save as NPZ (no images from Fritz — zeros)
            L = x.shape[0]
            images = np.zeros((3, 63, 63), dtype=np.float32)  # single dummy image

            np.savez_compressed(
                out_path,
                x=x,
                images=images,
                label=-1,
                group_id=group_id,
                group_name=group_name,
            )

            # Build manifest row with richer metadata
            classifications = "; ".join(src.get("classifications", []))
            cls_best = cls_info["classification"] if cls_info else classifications
            cls_prob = cls_info["probability"] if cls_info else ""
            cls_type = cls_info["classifier"] if cls_info else ""

            manifest_rows.append(
                {
                    "obj_id": obj_id,
                    "group_id": group_id,
                    "group_name": group_name,
                    "n_detections": L,
                    "classification": cls_best,
                    "classification_probability": cls_prob,
                    "classifier_type": cls_type,
                    "ra": src.get("ra", ""),
                    "dec": src.get("dec", ""),
                    "redshift": src.get("redshift", ""),
                }
            )
            total_preprocessed += 1

            if (idx + 1) % 20 == 0:
                print(
                    f"    [{idx + 1}/{len(sources)}] {total_preprocessed} sources preprocessed..."
                )

        print(f"  Group {group_name}: {total_preprocessed} total preprocessed so far")

    # Write manifest
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nManifest saved: {manifest_path} ({len(manifest_rows)} sources)")

    print("\n=== DONE ===")
    print(f"  Fetched: {total_fetched}")
    print(f"  Preprocessed: {total_preprocessed}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
