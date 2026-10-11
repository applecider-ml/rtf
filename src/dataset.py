"""
Standalone PyTorch datasets for ZTF alert data.

Four dataset classes:
  - ShardedDataset: loads from packed .pt shard files (fastest, from pack_shards.py)
  - MetaNPZDataset: loads from pre-processed NPZ files with metadata+images
  - PhotoNPZDataset: loads from pre-processed NPZ files (photo_events/)
  - AlertDataset: loads directly from raw alerts.npy files (data_ztf/)

ShardedDataset is preferred for large-scale training — it eliminates the
catastrophic GPFS file-open overhead of 1.74M individual np.load() calls.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

# Number of base continuous channels (dt, dt_prev, logflux, logflux_err)
N_CONT_BASE = 4

# Alert metadata fields to extract from each alert candidate.
# BOOM-compatible subset: only the 11 keys that exist in BOTH the public
# ZTF alert archive (ztf.uw.edu) AND the BOOM API (ZTF_alerts_aux).
# This ensures models trained on these keys work identically on both
# data sources without distribution shift from missing metadata.
# Original list had 30 keys (19 were PS1/color fields BOOM doesn't store).
# ORDER MUST MATCH the repacked shard column order (sorted by original
# position in the full 30-key list), NOT alphabetical!
ALERT_META_KEYS = [
    "sharpnr",  # sharpness of nearest source
    "scorr",  # detection significance (S/N)
    "diffmaglim",  # 5-sigma limiting mag
    "sky",  # local sky background
    "sigmapsf",  # PSF mag uncertainty
    "chinr",  # chi^2 of nearest ref source
    "rb",  # Real/Bogus score
    "chipsf",  # chi^2 of PSF fit
    "distnr",  # distance to nearest ref source (arcsec)
    "magnr",  # mag of nearest ref source
    "fwhm",  # PSF FWHM
]

# Total input channels: 4 base + 3 one-hot band + len(ALERT_META_KEYS) metadata
N_META = len(ALERT_META_KEYS)
IN_CHANNELS_BASE = 7  # dt, dt_prev, logflux, logflux_err, band_g, band_r, band_i
IN_CHANNELS_META = 7 + N_META  # base + alert metadata


class AlertDataset(Dataset):
    """
    Loads ZTF alerts directly from alerts.npy files in data_ztf/.

    Extracts per-observation:
      - Photometry: dt, dt_prev, logflux, logflux_err, one-hot band (7 channels)
      - Metadata: 28 alert candidate fields (ALERT_META_KEYS)

    Args:
        data_dir: path to data_ztf/ directory containing {obj_id}/alerts.npy
        splits_path: path to splits.csv (obj_id, split columns)
        split: one of "train", "val", "test"
        labels_path: path to directory containing {split}/{obj_id}.npz with labels
        use_metadata: if True, include alert metadata (35 channels total)
        max_len: maximum sequence length
        horizon: maximum dt in days (None = no cut)
    """

    COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 4}  # noqa: RUF012

    def __init__(
        self,
        data_dir: str,
        splits_path: str,
        split: str,
        labels_path: str,
        use_metadata: bool = True,
        max_len: int = 257,
        horizon: float | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.labels_path = Path(labels_path)
        self.use_metadata = use_metadata
        self.in_channels = IN_CHANNELS_META if use_metadata else IN_CHANNELS_BASE
        self.max_len = max_len
        self.horizon = horizon

        # Load split assignments
        self.obj_ids = []
        with open(splits_path) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    oid = row["obj_id"]
                    alerts_path = self.data_dir / oid / "alerts.npy"
                    if alerts_path.exists():
                        self.obj_ids.append(oid)

        if not self.obj_ids:
            raise FileNotFoundError(f"No sources found for split={split}")

        # Pre-load labels from NPZ files
        self.labels = {}
        for oid in self.obj_ids:
            label_file = self.labels_path / split / f"{oid}.npz"
            if label_file.exists():
                npz = np.load(label_file, allow_pickle=True)
                self.labels[oid] = int(npz["label"])
            else:
                self.labels[oid] = -1  # unknown

        # Compute normalization stats lazily on first access
        self._norm_mean = None
        self._norm_std = None

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception:
            x = torch.zeros(1, self.in_channels)
            return {
                "x": x,
                "label_fine": 0,
                "label_coarse": 0,
                "obj_id": "BAD",
                "seq_len": 1,
            }

    def _load_item(self, idx):
        oid = self.obj_ids[idx]
        alerts = np.load(self.data_dir / oid / "alerts.npy", allow_pickle=True)
        label = self.labels.get(oid, 0)

        # Sort by JD
        jds = np.array([a["candidate"]["jd"] for a in alerts])
        order = np.argsort(jds)
        alerts = alerts[order]
        jds = jds[order]

        L = len(alerts)
        if L == 0:
            x = torch.zeros(1, self.in_channels)
            return {
                "x": x,
                "label_fine": label,
                "label_coarse": self.COARSE_MAP.get(label, 0),
                "obj_id": oid,
                "seq_len": 1,
            }

        # Time features
        dt = (jds - jds[0]).astype(np.float32)
        dt_prev = np.zeros(L, dtype=np.float32)
        dt_prev[1:] = np.diff(jds).astype(np.float32)

        # Horizon cut
        if self.horizon is not None:
            mask = dt <= self.horizon
            alerts = alerts[mask]
            dt = dt[mask]
            dt_prev = dt_prev[mask]
            L = len(alerts)

        # Photometry
        fids = np.array([a["candidate"]["fid"] for a in alerts])
        magpsf = np.array([a["candidate"]["magpsf"] for a in alerts], dtype=np.float32)
        sigmapsf = np.array(
            [a["candidate"]["sigmapsf"] for a in alerts], dtype=np.float32
        )

        # Convert mag to logflux: logflux = -0.4 * magpsf + const (we drop const, relative is fine)
        logflux = (-0.4 * magpsf).astype(np.float32)
        logflux_err = (0.4 * sigmapsf * np.log(10) / np.log(10)).astype(
            np.float32
        )  # ~0.4 * sigmapsf

        # Log-transform time
        log_dt = np.log1p(dt)
        log_dt_prev = np.log1p(dt_prev)

        # One-hot band (fid: 1=g, 2=r, 3=i → index 0, 1, 2)
        band_idx = (fids - 1).clip(0, 2).astype(np.int64)
        one_hot = np.eye(3, dtype=np.float32)[band_idx]

        # Base features: (L, 7)
        base = np.column_stack([log_dt, log_dt_prev, logflux, logflux_err])
        x = np.concatenate([base, one_hot], axis=1)

        if self.use_metadata:
            # Extract alert metadata
            meta = np.zeros((L, N_META), dtype=np.float32)
            for i, a in enumerate(alerts):
                cand = a["candidate"]
                for j, key in enumerate(ALERT_META_KEYS):
                    val = cand.get(key, np.nan)
                    if val is None or (isinstance(val, (int, float)) and val == -999):
                        val = np.nan
                    try:
                        meta[i, j] = float(val)
                    except (ValueError, TypeError):
                        meta[i, j] = np.nan

            # Replace NaN with 0 for metadata (model learns to use validity from context)
            meta = np.nan_to_num(meta, nan=0.0)
            x = np.concatenate([x, meta], axis=1)

        # Truncate
        L = min(x.shape[0], self.max_len)
        x = x[:L]

        return {
            "x": torch.from_numpy(x),
            "label_fine": label,
            "label_coarse": self.COARSE_MAP.get(label, 0),
            "obj_id": oid,
            "seq_len": L,
        }


# ---------------------------------------------------------------------------
# Pre-processed metadata NPZ dataset (from preprocess_alerts.py)
# ---------------------------------------------------------------------------


class MetaNPZDataset(Dataset):
    """
    Loads pre-processed NPZ files with photometry + metadata + images + GP features.

    Each NPZ contains:
        x:           (L, 37) float32 — sequence features
        images:      (L, 3, 63, 63) float32 — cutout stamps (if available)
        has_image:   (L,) float32 — 1.0 where image is valid
        gp_features: (N_GP,) float32 — GP features from lightcurve-fitting (if available)
        gp_keys:     string array — feature names (for reference)
        label:       int64

    Created by preprocess_alerts.py from raw alerts.npy files.
    """

    COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 4}  # noqa: RUF012

    def __init__(
        self,
        data_dir: str,
        max_len: int = 257,
        use_images: bool = False,
        use_gp: bool = False,
        max_detections: int | None = None,
        random_truncate: bool = False,
        min_detections: int = 3,
    ):
        self.use_gp = use_gp
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.use_images = use_images
        self.max_detections = max_detections
        self.random_truncate = random_truncate
        self.min_detections = min_detections
        self.in_channels = IN_CHANNELS_META

        # NOTE: We skip the f.stat() filesize check here because on GPFS,
        # calling stat() on 200k+ files per directory causes a multi-hour
        # network metadata deadlock. Our preprocess_alerts.py already
        # validated every file, so there are no corrupt files to filter.
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No valid NPZ files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception:
            x = torch.zeros(1, self.in_channels)
            result = {
                "x": x,
                "label_fine": 0,
                "label_coarse": 0,
                "obj_id": "BAD",
                "seq_len": 1,
            }
            if self.use_images:
                result["image"] = torch.zeros(3, 63, 63)
            if self.use_gp:
                result["gp_features"] = torch.zeros(
                    1
                )  # placeholder, real size set at collate
            return result

    def _load_item(self, idx):
        npz = np.load(self.files[idx], allow_pickle=True)
        x = npz["x"].astype(np.float32)
        label = int(npz["label"])

        L = min(x.shape[0], self.max_len)
        if self.max_detections is not None:
            L = min(L, self.max_detections)
        if self.random_truncate and L > self.min_detections:
            # Randomly truncate to [min_detections, L] observations
            # Simulates seeing the source at different "ages"
            import random

            L = random.randint(self.min_detections, L)
        x = x[:L]

        # Normalize metadata channels to prevent gradient explosion.
        # Base channels (0:4) are log-transformed, (4:7) are one-hot.
        # Metadata channels (7:) have mixed scales (0-300+), so we
        # apply per-feature robust scaling within each sample.
        if x.shape[1] > 7:
            meta = x[:, 7:]
            # Per-feature: center on median, scale by IQR (robust to outliers)
            for j in range(meta.shape[1]):
                col = meta[:, j]
                med = np.median(col)
                iqr = np.percentile(col, 75) - np.percentile(col, 25)
                if iqr > 1e-6:
                    meta[:, j] = (col - med) / iqr
                else:
                    meta[:, j] = col - med
            # Final clip for safety
            x[:, 7:] = np.clip(meta, -10.0, 10.0)

        result = {
            "x": torch.from_numpy(x),
            "label_fine": label,
            "label_coarse": self.COARSE_MAP.get(label, 0),
            "obj_id": self.files[idx].stem,
            "seq_len": L,
        }

        if self.use_images and "images" in npz:
            all_images = npz["images"][:L].astype(np.float32)
            all_has_image = npz["has_image"][:L].astype(np.float32)
            # Use the most recent stamp with a valid image (matches alert packet)
            valid_idx = np.where(all_has_image > 0)[0]
            if len(valid_idx) > 0:
                img = all_images[valid_idx[-1]]  # most recent valid stamp (3, 63, 63)
                # Normalize per-channel
                for c in range(3):
                    ch = img[c].astype(np.float64)
                    p1, p99 = np.percentile(ch, [1, 99])
                    ch = np.clip(ch, p1, p99)
                    std = ch.std()
                    if std > 1e-6:
                        img[c] = ((ch - ch.mean()) / std).astype(np.float32)
                    else:
                        img[c] = 0.0
                result["image"] = torch.from_numpy(img)
            else:
                result["image"] = torch.zeros(3, 63, 63)
        elif self.use_images:
            result["image"] = torch.zeros(3, 63, 63)

        if self.use_gp and "gp_features" in npz:
            gp = npz["gp_features"].astype(np.float32)
            gp_keys = list(npz["gp_keys"]) if "gp_keys" in npz else []
            gp = np.nan_to_num(gp, nan=0.0, posinf=0.0, neginf=0.0)
            gp = np.clip(gp, -10.0, 10.0)
            # Map to canonical 114-dim vector by key name
            if len(gp_keys) == 114:
                result["gp_features"] = torch.from_numpy(gp)
            else:
                # Build canonical key→index mapping on first call
                if not hasattr(self, "_canonical_gp_keys"):
                    self._canonical_gp_keys = None
                    # Find a source with 114 keys to establish canonical order
                    for f in self.files[:100]:
                        n = np.load(f, allow_pickle=True)
                        if "gp_keys" in n and len(n["gp_keys"]) == 114:
                            self._canonical_gp_keys = {
                                k: i for i, k in enumerate(n["gp_keys"])
                            }
                            break
                gp_padded = np.zeros(114, dtype=np.float32)
                if self._canonical_gp_keys is not None:
                    for j, key in enumerate(gp_keys):
                        if key in self._canonical_gp_keys:
                            gp_padded[self._canonical_gp_keys[key]] = gp[j]
                else:
                    gp_padded[: len(gp)] = gp  # fallback
                result["gp_features"] = torch.from_numpy(gp_padded)
        elif self.use_gp:
            result["gp_features"] = torch.zeros(114)

        return result


# ---------------------------------------------------------------------------
# Sharded dataset for large-scale training (from pack_shards.py)
# Streams one shard at a time — uses ~1GB peak RAM instead of 80GB.
# ---------------------------------------------------------------------------


class ShardedDataset(IterableDataset):
    """Streaming dataset that loads one .pt shard at a time.

    Instead of loading all 88 shards (~80GB) into RAM upfront, this streams
    through shards sequentially, loading and freeing each one. Shuffling is
    done at two levels: shard order is randomized, and samples within each
    shard are shuffled.

    Peak RAM usage: ~1GB (one shard) instead of ~80GB (all shards).
    """

    COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 4}  # noqa: RUF012

    def __init__(
        self,
        shards_dir: str,
        max_len: int = 257,
        use_images: bool = False,
        use_gp: bool = False,
        gp_dim: int = 80,
        max_detections: int | None = None,
        random_truncate: bool = False,
        min_detections: int = 3,
    ):
        super().__init__()
        self.shards_dir = str(shards_dir)
        self.max_len = max_len
        self.use_images = use_images
        self.use_gp = use_gp
        self.gp_dim = gp_dim
        self.max_detections = max_detections
        self.random_truncate = random_truncate
        self.min_detections = min_detections

        shard_files = sorted(Path(shards_dir).glob("shard_*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files in {shards_dir}")
        self.shard_files = [str(f) for f in shard_files]

        # Auto-detect in_channels from the first shard's tensor width.
        # This supports both old 18-channel ZTF shards (4 base + 3 band + 11 meta)
        # and new 21-channel joint shards (4 base + 6 band + 11 meta).
        peek = torch.load(self.shard_files[0], weights_only=False, map_location="cpu")
        self.in_channels = peek["x_concat"].shape[1]
        # Metadata columns start after base + band one-hot channels.
        # We know there are always N_META (11) metadata columns at the end,
        # so the non-meta prefix = in_channels - N_META.
        self.n_base_channels = self.in_channels - N_META
        del peek
        print(f"  [SHARDS] Auto-detected {self.in_channels} channels "
              f"(base+band={self.n_base_channels}, meta={N_META})")

        # Estimate total samples without loading any shards.
        # pack_shards.py writes exactly SHARD_SIZE (20,000) samples per shard,
        # except the last shard which has the remainder.
        # The exact count only matters for tqdm/progress bars; __iter__
        # yields the actual samples regardless.
        shard_size = 20_000
        self._total = shard_size * (len(self.shard_files) - 1) + shard_size  # upper bound
        print(f"  [SHARDS] ~{self._total:,} samples across "
              f"{len(self.shard_files)} shards (streaming mode, ~1GB peak RAM)")

    def __len__(self):
        return self._total

    def __iter__(self):
        # Determine which shards this worker handles
        worker_info = torch.utils.data.get_worker_info()
        shard_indices = list(range(len(self.shard_files)))

        if worker_info is not None:
            # Split shards across workers (no duplicates)
            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            shard_indices = [i for i in shard_indices
                            if i % n_workers == worker_id]

        # Shuffle shard order for this epoch
        random.shuffle(shard_indices)

        yielded = 0
        for si, shard_idx in enumerate(shard_indices):
            sf = self.shard_files[shard_idx]
            shard = torch.load(sf, weights_only=False, map_location="cpu")
            n = len(shard["labels"])

            # Print progress (only from worker 0 or main process)
            if (worker_info is None or worker_info.id == 0) and ((si + 1) % 5 == 0 or si == 0):
                    print(f"    [shard {si+1}/{len(shard_indices)}] "
                          f"{yielded:,} samples streamed so far")

            # Shuffle sample order within this shard
            indices = list(range(n))
            random.shuffle(indices)

            for local_idx in indices:
                sample = self._process_sample(shard, local_idx)
                if sample is not None:
                    yielded += 1
                    yield sample

            # Free memory immediately
            del shard

        if worker_info is None or worker_info.id == 0:
            print(f"    [DONE] {yielded:,} samples streamed this epoch")

    def _process_sample(self, shard, local_idx):
        try:
            # Extract raw sequence from concatenated tensor
            start = shard["x_offsets"][local_idx].item()
            end = shard["x_offsets"][local_idx + 1].item()
            x = shard["x_concat"][start:end].numpy().copy()
            label = shard["labels"][local_idx].item()
            obj_id = shard["obj_ids"][local_idx]

            # Apply truncation (identical to MetaNPZDataset._load_item)
            L = min(x.shape[0], self.max_len)
            if self.max_detections is not None:
                L = min(L, self.max_detections)
            if self.random_truncate and L > self.min_detections:
                L = random.randint(self.min_detections, L)
            x = x[:L]

            # IQR normalization on metadata columns.
            # n_base_channels is auto-detected: 7 for old ZTF shards, 10 for joint.
            nb = self.n_base_channels
            if x.shape[1] > nb:
                meta = x[:, nb:]
                for j in range(meta.shape[1]):
                    col = meta[:, j]
                    med = np.median(col)
                    iqr = np.percentile(col, 75) - np.percentile(col, 25)
                    if iqr > 1e-6:
                        meta[:, j] = (col - med) / iqr
                    else:
                        meta[:, j] = col - med
                x[:, nb:] = np.clip(meta, -10.0, 10.0)

            result = {
                "x": torch.from_numpy(x),
                "label_fine": label,
                "label_coarse": self.COARSE_MAP.get(label, 0),
                "obj_id": obj_id,
                "seq_len": L,
            }

            if self.use_images:
                result["image"] = shard["images"][local_idx]

            if self.use_gp:
                if "gp_features" in shard and shard["gp_features"].shape[-1] > 0:
                    gp = shard["gp_features"][local_idx].float().clone()
                    # Handle dimension mismatch: pad or truncate to gp_dim
                    actual_dim = gp.shape[0]
                    if actual_dim == self.gp_dim:
                        pass  # perfect match
                    elif actual_dim < self.gp_dim:
                        padded = torch.zeros(self.gp_dim)
                        padded[:actual_dim] = gp
                        gp = padded
                    else:
                        gp = gp[:self.gp_dim]
                    # Clean up non-finite values
                    gp = torch.nan_to_num(gp, nan=0.0, posinf=0.0, neginf=0.0)
                    gp = gp.clamp(-10.0, 10.0)
                    result["gp_features"] = gp
                else:
                    # Shards without GP features: return zeros
                    result["gp_features"] = torch.zeros(self.gp_dim)

            return result

        except Exception:
            return None


# ---------------------------------------------------------------------------
# Legacy dataset for pre-processed NPZ files (photometry only)
# ---------------------------------------------------------------------------


class PhotoNPZDataset(Dataset):
    """
    Loads ZTF photometry sequences from pre-processed NPZ files.

    Each NPZ contains:
        data:    (L, 15) float32
        columns: string array
        label:   int64 scalar

    Args:
        use_metadata: If True, include inter-band colors from NPZ (13 channels).
    """

    COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 4}  # noqa: RUF012

    def __init__(
        self,
        data_dir: str,
        stats_path: str | None = None,
        max_len: int = 257,
        horizon: float | None = None,
        use_metadata: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.horizon = horizon
        self.use_metadata = use_metadata
        self.in_channels = 13 if use_metadata else 7

        candidates = sorted(self.data_dir.glob("*.npz"))
        self.files = [f for f in candidates if f.stat().st_size >= 200]
        n_bad = len(candidates) - len(self.files)
        if not self.files:
            raise FileNotFoundError(f"No valid NPZ files in {data_dir}")
        if n_bad > 0:
            print(f"  Filtered out {n_bad} too-small NPZ files from {data_dir}")

        if stats_path is not None:
            st = np.load(stats_path)
            self.mean = st["mean"].flatten()[:N_CONT_BASE].astype(np.float32)
            self.std = st["std"].flatten()[:N_CONT_BASE].astype(np.float32)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception:
            x = torch.zeros(1, self.in_channels)
            return {
                "x": x,
                "label_fine": 0,
                "label_coarse": 0,
                "obj_id": "BAD",
                "seq_len": 1,
            }

    def _load_item(self, idx):
        npz = np.load(self.files[idx], allow_pickle=True)
        data = npz["data"]
        label = int(npz["label"])

        if self.horizon is not None:
            data = data[data[:, 0] <= self.horizon]

        dt = np.log1p(data[:, 0]).astype(np.float32)
        dt_prev = np.log1p(data[:, 1]).astype(np.float32)
        logf = data[:, 3].astype(np.float32)
        logfe = data[:, 4].astype(np.float32)
        band = data[:, 2].astype(np.int64)

        cont = np.stack([dt, dt_prev, logf, logfe], axis=1)
        one_hot = np.eye(3, dtype=np.float32)[band]

        if self.mean is not None:
            cont = (cont - self.mean) / (self.std + 1e-8)

        if self.use_metadata and data.shape[1] >= 14:
            colors = np.nan_to_num(data[:, 8:12].astype(np.float32), nan=0.0)
            flags = data[:, 12:14].astype(np.float32)
            x = np.concatenate([cont, one_hot, colors, flags], axis=1)
        else:
            x = np.concatenate([cont, one_hot], axis=1)

        L = min(x.shape[0], self.max_len)
        x = x[:L]

        return {
            "x": torch.from_numpy(x),
            "label_fine": label,
            "label_coarse": self.COARSE_MAP[label],
            "obj_id": self.files[idx].stem,
            "seq_len": L,
        }


# ---------------------------------------------------------------------------
# Collate function (shared by both datasets)
# ---------------------------------------------------------------------------


def collate_fn(batch):
    """Pad variable-length sequences to max length in batch."""
    max_len = max(item["seq_len"] for item in batch)
    B = len(batch)
    C = batch[0]["x"].shape[1]
    has_image = "image" in batch[0]

    x_padded = torch.zeros(B, max_len, C)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    labels_fine = torch.zeros(B, dtype=torch.long)
    labels_coarse = torch.zeros(B, dtype=torch.long)
    obj_ids = []

    if has_image:
        images = torch.zeros(B, 3, 63, 63)

    for i, item in enumerate(batch):
        L = item["seq_len"]
        x_padded[i, :L] = item["x"]
        pad_mask[i, :L] = False
        labels_fine[i] = item["label_fine"]
        labels_coarse[i] = item["label_coarse"]
        obj_ids.append(item["obj_id"])
        if has_image:
            images[i] = item["image"]

    result = {
        "x": x_padded,
        "pad_mask": pad_mask,
        "label_fine": labels_fine,
        "label_coarse": labels_coarse,
        "obj_ids": obj_ids,
    }
    if has_image:
        result["images"] = images

    # GP features: static per-source vector, pad to max GP dim in batch
    if "gp_features" in batch[0]:
        gp_dim = max(item["gp_features"].shape[0] for item in batch)
        gp_padded = torch.zeros(B, gp_dim)
        for i, item in enumerate(batch):
            gp = item["gp_features"]
            gp_padded[i, : gp.shape[0]] = gp
        result["gp_features"] = gp_padded

    return result
