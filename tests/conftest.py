"""Shared fixtures for alert-compression tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_batch():
    """A minimal batch matching the collate_fn output format (7 channels)."""
    B, L = 4, 20
    x = torch.randn(B, L, 7)
    x[..., 4:7] = 0.0
    bands = torch.randint(0, 3, (B, L))
    for b in range(B):
        for t in range(L):
            x[b, t, 4 + bands[b, t]] = 1.0
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -5:] = True
    return {
        "x": x,
        "pad_mask": pad_mask,
        "label_fine": torch.randint(0, 10, (B,)),
        "label_coarse": torch.randint(0, 5, (B,)),
        "obj_ids": [f"ZTF_test_{i}" for i in range(B)],
    }


@pytest.fixture
def sample_batch_images():
    """A batch with images + metadata (37 channels)."""
    B, L, C = 4, 15, 37
    x = torch.randn(B, L, C)
    x[..., 4:7] = 0.0
    bands = torch.randint(0, 3, (B, L))
    for b in range(B):
        for t in range(L):
            x[b, t, 4 + bands[b, t]] = 1.0
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -3:] = True
    return {
        "x": x,
        "pad_mask": pad_mask,
        "label_fine": torch.randint(0, 10, (B,)),
        "label_coarse": torch.randint(0, 5, (B,)),
        "obj_ids": [f"ZTF_test_{i}" for i in range(B)],
        "images": torch.randn(B, 3, 63, 63) * 0.1,
    }


@pytest.fixture
def sample_batch_gp():
    """A batch with GP features + metadata (37 channels)."""
    B, L, C = 4, 15, 37
    x = torch.randn(B, L, C)
    x[..., 4:7] = 0.0
    bands = torch.randint(0, 3, (B, L))
    for b in range(B):
        for t in range(L):
            x[b, t, 4 + bands[b, t]] = 1.0
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -3:] = True
    return {
        "x": x,
        "pad_mask": pad_mask,
        "label_fine": torch.randint(0, 10, (B,)),
        "label_coarse": torch.randint(0, 5, (B,)),
        "obj_ids": [f"ZTF_test_{i}" for i in range(B)],
        "gp_features": torch.randn(B, 12),
    }


@pytest.fixture
def sample_batch_all():
    """A batch with all modalities: images + GP + metadata."""
    B, L, C = 4, 15, 37
    x = torch.randn(B, L, C)
    x[..., 4:7] = 0.0
    bands = torch.randint(0, 3, (B, L))
    for b in range(B):
        for t in range(L):
            x[b, t, 4 + bands[b, t]] = 1.0
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, -3:] = True
    return {
        "x": x,
        "pad_mask": pad_mask,
        "label_fine": torch.randint(0, 10, (B,)),
        "label_coarse": torch.randint(0, 5, (B,)),
        "obj_ids": [f"ZTF_test_{i}" for i in range(B)],
        "images": torch.randn(B, 3, 63, 63) * 0.1,
        "gp_features": torch.randn(B, 12),
    }


@pytest.fixture
def fake_npz_dir(tmp_path):
    """Create a temp directory with synthetic NPZ files matching the ZTF format."""
    n_files = 10
    for i in range(n_files):
        L = np.random.randint(5, 50)
        data = np.zeros((L, 7), dtype=np.float32)
        data[:, 0] = np.sort(np.random.uniform(0, 100, L))
        data[:, 1] = np.random.uniform(0, 5, L)
        data[:, 2] = np.random.randint(0, 3, L).astype(np.float32)
        data[:, 3] = np.random.uniform(-1, 2, L)
        data[:, 4] = np.random.uniform(0.01, 0.3, L)
        label = np.int64(np.random.randint(0, 10))
        np.savez(
            tmp_path / f"ZTF_fake_{i:03d}.npz",
            data=data,
            columns=np.array(
                ["dt", "dt_prev", "band_id", "logflux", "logflux_err", "c5", "c6"]
            ),
            label=label,
        )
    return tmp_path


@pytest.fixture
def fake_meta_npz_dir(tmp_path):
    """Create synthetic MetaNPZ files with images + GP features."""
    out_dir = tmp_path / "meta_npz"
    out_dir.mkdir()
    for i in range(10):
        L = np.random.randint(5, 30)
        x = np.random.randn(L, 37).astype(np.float32)
        x[:, 4:7] = 0.0
        for t in range(L):
            x[t, 4 + np.random.randint(0, 3)] = 1.0
        label = np.int64(np.random.randint(0, 10))
        images = np.random.randn(L, 3, 63, 63).astype(np.float32) * 0.1
        has_image = np.ones(L, dtype=np.float32)
        gp_features = np.random.randn(12).astype(np.float32)
        np.savez(
            out_dir / f"ZTF_meta_{i:03d}.npz",
            x=x,
            label=label,
            images=images,
            has_image=has_image,
            gp_features=gp_features,
            gp_keys=np.array([f"gp_{j}" for j in range(12)]),
        )
    return out_dir


@pytest.fixture
def fake_stats(tmp_path):
    """Create a fake feature_stats file."""
    stats_path = tmp_path / "feature_stats_day100.npz"
    np.savez(
        stats_path,
        mean=np.zeros(4, dtype=np.float32),
        std=np.ones(4, dtype=np.float32),
    )
    return str(stats_path)
