"""
Standalone PyTorch datasets for ZTF alert data.

Two dataset classes:
  - PhotoNPZDataset: loads from pre-processed NPZ files (photo_events/)
  - AlertDataset: loads directly from raw alerts.npy files (data_ztf/),
    extracting photometry + 36 alert metadata fields per observation

AlertDataset is preferred — it uses the raw alert data with no intermediate
preprocessing, and includes rich metadata (star/galaxy scores, PSF shape,
real/bogus, nearest source properties, etc.) that improve compression quality.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Number of base continuous channels (dt, dt_prev, logflux, logflux_err)
N_CONT_BASE = 4

# Alert metadata fields to extract from each alert candidate.
# These are the most informative fields for transient classification.
ALERT_META_KEYS = [
    "sgscore1", "sgscore2",      # star/galaxy scores for PS1 neighbors
    "distpsnr1", "distpsnr2",    # distance to nearest PS1 sources (arcsec)
    "nmtchps",                    # number of PS1 matches
    "sharpnr",                    # sharpness of nearest source
    "scorr",                      # detection significance (S/N)
    "diffmaglim",                 # 5-sigma limiting mag
    "sky",                        # local sky background
    "ndethist",                   # cumulative detections
    "ncovhist",                   # times field observed
    "sigmapsf",                   # PSF mag uncertainty
    "chinr",                      # chi^2 of nearest PS1 source
    "classtar",                   # SExtractor star/galaxy
    "rb",                         # Real/Bogus score
    "chipsf",                     # chi^2 of PSF fit
    "distnr",                     # distance to nearest ref source (arcsec)
    "magnr",                      # mag of nearest ref source
    "fwhm",                       # PSF FWHM
    "srmag1", "sgmag1", "simag1", "szmag1",  # PS1 mags of nearest source
    "srmag2", "sgmag2", "simag2", "szmag2",  # PS1 mags of 2nd nearest
    "clrcoeff", "clrcounc",      # color coefficients
    "zpclrcov",                   # zeropoint color covariance
]

# Total input channels: 4 base + 3 one-hot band + len(ALERT_META_KEYS) metadata
N_META = len(ALERT_META_KEYS)
IN_CHANNELS_BASE = 7       # dt, dt_prev, logflux, logflux_err, band_g, band_r, band_i
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

    COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 4}

    def __init__(
        self,
        data_dir: str,
        splits_path: str,
        split: str,
        labels_path: str,
        use_metadata: bool = True,
        max_len: int = 257,
        horizon: float = None,
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
            return {"x": x, "label_fine": 0, "label_coarse": 0,
                    "obj_id": "BAD", "seq_len": 1}

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
            return {"x": x, "label_fine": label, "label_coarse": self.COARSE_MAP.get(label, 0),
                    "obj_id": oid, "seq_len": 1}

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
        sigmapsf = np.array([a["candidate"]["sigmapsf"] for a in alerts], dtype=np.float32)

        # Convert mag to logflux: logflux = -0.4 * magpsf + const (we drop const, relative is fine)
        logflux = (-0.4 * magpsf).astype(np.float32)
        logflux_err = (0.4 * sigmapsf * np.log(10) / np.log(10)).astype(np.float32)  # ~0.4 * sigmapsf

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
    Loads pre-processed NPZ files with photometry + metadata.

    Each NPZ contains:
        x:     (L, 37) float32 — ready for model input
        label: int64

    Created by preprocess_alerts.py from raw alerts.npy files.
    """

    COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 4}

    def __init__(self, data_dir: str, max_len: int = 257):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.in_channels = IN_CHANNELS_META

        candidates = sorted(list(self.data_dir.glob("*.npz")))
        self.files = [f for f in candidates if f.stat().st_size >= 200]
        if not self.files:
            raise FileNotFoundError(f"No valid NPZ files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception:
            x = torch.zeros(1, self.in_channels)
            return {"x": x, "label_fine": 0, "label_coarse": 0,
                    "obj_id": "BAD", "seq_len": 1}

    def _load_item(self, idx):
        npz = np.load(self.files[idx], allow_pickle=True)
        x = npz["x"].astype(np.float32)
        label = int(npz["label"])

        L = min(x.shape[0], self.max_len)
        x = x[:L]

        return {
            "x": torch.from_numpy(x),
            "label_fine": label,
            "label_coarse": self.COARSE_MAP.get(label, 0),
            "obj_id": self.files[idx].stem,
            "seq_len": L,
        }


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

    COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 4}

    def __init__(
        self,
        data_dir: str,
        stats_path: str = None,
        max_len: int = 257,
        horizon: float = None,
        use_metadata: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.horizon = horizon
        self.use_metadata = use_metadata
        self.in_channels = 13 if use_metadata else 7

        candidates = sorted(list(self.data_dir.glob("*.npz")))
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
            return {"x": x, "label_fine": 0, "label_coarse": 0,
                    "obj_id": "BAD", "seq_len": 1}

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

    x_padded = torch.zeros(B, max_len, C)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    labels_fine = torch.zeros(B, dtype=torch.long)
    labels_coarse = torch.zeros(B, dtype=torch.long)
    obj_ids = []

    for i, item in enumerate(batch):
        L = item["seq_len"]
        x_padded[i, :L] = item["x"]
        pad_mask[i, :L] = False
        labels_fine[i] = item["label_fine"]
        labels_coarse[i] = item["label_coarse"]
        obj_ids.append(item["obj_id"])

    return {
        "x": x_padded,
        "pad_mask": pad_mask,
        "label_fine": labels_fine,
        "label_coarse": labels_coarse,
        "obj_ids": obj_ids,
    }
