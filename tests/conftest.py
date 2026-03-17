"""Shared fixtures for alert-compression tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_batch():
    """A minimal batch matching the collate_fn output format."""
    B, L = 4, 20
    x = torch.randn(B, L, 7)
    # Make band channels valid one-hot
    x[..., 4:7] = 0.0
    bands = torch.randint(0, 3, (B, L))
    for b in range(B):
        for t in range(L):
            x[b, t, 4 + bands[b, t]] = 1.0
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    # Last 5 positions are padding for each sample
    pad_mask[:, -5:] = True
    return {
        "x": x,
        "pad_mask": pad_mask,
        "label_fine": torch.randint(0, 10, (B,)),
        "label_coarse": torch.randint(0, 5, (B,)),
        "obj_ids": [f"ZTF_test_{i}" for i in range(B)],
    }


@pytest.fixture
def fake_npz_dir(tmp_path):
    """Create a temp directory with synthetic NPZ files matching the ZTF format."""
    n_files = 10
    for i in range(n_files):
        L = np.random.randint(5, 50)
        # Columns: dt, dt_prev, band_id, logflux, logflux_err, (+ extras ignored)
        data = np.zeros((L, 7), dtype=np.float32)
        data[:, 0] = np.sort(np.random.uniform(0, 100, L))  # dt
        data[:, 1] = np.random.uniform(0, 5, L)  # dt_prev
        data[:, 2] = np.random.randint(0, 3, L).astype(np.float32)  # band_id
        data[:, 3] = np.random.uniform(-1, 2, L)  # logflux
        data[:, 4] = np.random.uniform(0.01, 0.3, L)  # logflux_err
        label = np.int64(np.random.randint(0, 10))
        np.savez(tmp_path / f"ZTF_fake_{i:03d}.npz", data=data,
                 columns=np.array(["dt", "dt_prev", "band_id", "logflux", "logflux_err", "c5", "c6"]),
                 label=label)
    return tmp_path


@pytest.fixture
def fake_stats(tmp_path):
    """Create a fake feature_stats file."""
    stats_path = tmp_path / "feature_stats_day100.npz"
    np.savez(stats_path, mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
    return str(stats_path)
