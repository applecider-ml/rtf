"""
Standalone PyTorch dataset for ZTF photometry NPZ files.
No Hyrax dependency — reads the same NPZ format as applecider.datasets.photo_dataset.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PhotoNPZDataset(Dataset):
    """
    Loads ZTF photometry sequences from NPZ files.

    Each NPZ contains:
        data:    (L, 7) float32 — [dt, dt_prev, band_id, logflux, logflux_err, ...]
        columns: string array
        label:   int64 scalar

    We transform to the model input format:
        [log1p(dt), log1p(dt_prev), logflux, logflux_err, one_hot_band(3)]
    """

    # Fine → coarse class mapping
    COARSE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 4}

    def __init__(
        self,
        data_dir: str,
        stats_path: str = None,
        max_len: int = 257,
        horizon: float = None,
    ):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.horizon = horizon

        # Filter out obviously corrupt files (fast: stat only, no file open)
        candidates = sorted(list(self.data_dir.glob("*.npz")))
        self.files = [f for f in candidates if f.stat().st_size >= 200]
        n_bad = len(candidates) - len(self.files)
        if not self.files:
            raise FileNotFoundError(f"No valid NPZ files in {data_dir}")
        if n_bad > 0:
            print(f"  Filtered out {n_bad} too-small NPZ files from {data_dir}")

        # Load global normalization stats
        if stats_path is not None:
            st = np.load(stats_path)
            self.mean = st["mean"].astype(np.float32)
            self.std = st["std"].astype(np.float32)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception:
            # Return a minimal valid sample so the batch doesn't crash
            x = torch.zeros(1, 7)
            return {"x": x, "label_fine": 0, "label_coarse": 0,
                    "obj_id": "BAD", "seq_len": 1}

    def _load_item(self, idx):
        npz = np.load(self.files[idx], allow_pickle=True)
        data = npz["data"]  # (L, 7) raw format
        label = int(npz["label"])

        # Horizon cut if specified
        if self.horizon is not None:
            data = data[data[:, 0] <= self.horizon]

        # Extract features
        dt = np.log1p(data[:, 0]).astype(np.float32)
        dt_prev = np.log1p(data[:, 1]).astype(np.float32)
        logf = data[:, 3].astype(np.float32)
        logfe = data[:, 4].astype(np.float32)
        band = data[:, 2].astype(np.int64)

        # Stack continuous + one-hot band → (L, 7)
        cont = np.stack([dt, dt_prev, logf, logfe], axis=1)
        one_hot = np.eye(3, dtype=np.float32)[band]
        x = np.concatenate([cont, one_hot], axis=1)

        # Normalize continuous channels
        if self.mean is not None:
            x[:, :4] = (x[:, :4] - self.mean) / (self.std + 1e-8)

        # Truncate
        L = min(x.shape[0], self.max_len)
        x = x[:L]

        return {
            "x": torch.from_numpy(x),
            "label_fine": label,
            "label_coarse": self.COARSE_MAP[label],
            "obj_id": self.files[idx].stem,
            "seq_len": L,
        }


def collate_fn(batch):
    """Pad variable-length sequences to max length in batch."""
    max_len = max(item["seq_len"] for item in batch)
    B = len(batch)

    x_padded = torch.zeros(B, max_len, 7)
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
