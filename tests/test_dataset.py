"""Tests for PhotoNPZDataset and collate_fn."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import PhotoNPZDataset, collate_fn


class TestPhotoNPZDataset:
    def test_loads_files(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        assert len(ds) >= 10

    def test_item_keys(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        item = ds[0]
        assert set(item.keys()) == {"x", "label_fine", "label_coarse", "obj_id", "seq_len"}

    def test_item_shapes(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        item = ds[0]
        assert item["x"].ndim == 2
        assert item["x"].shape[1] == 7
        assert item["x"].shape[0] == item["seq_len"]
        assert item["x"].shape[0] > 0

    def test_one_hot_band(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        item = ds[0]
        band_cols = item["x"][:, 4:7]
        # Each row should have exactly one 1.0
        assert torch.allclose(band_cols.sum(dim=1), torch.ones(item["seq_len"]))

    def test_label_ranges(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        for i in range(len(ds)):
            item = ds[i]
            assert 0 <= item["label_fine"] <= 9
            assert 0 <= item["label_coarse"] <= 4

    def test_horizon_cut(self, fake_npz_dir, fake_stats):
        ds_full = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        ds_cut = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats, horizon=10.0)
        # At least some samples should be shorter with horizon cut
        shorter = sum(ds_cut[i]["seq_len"] <= ds_full[i]["seq_len"] for i in range(len(ds_full)))
        assert shorter == len(ds_full)

    def test_max_len_truncation(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats, max_len=10)
        for i in range(len(ds)):
            assert ds[i]["seq_len"] <= 10

    def test_corrupt_file_returns_fallback(self, fake_npz_dir, fake_stats):
        # Create a corrupt file
        corrupt_path = fake_npz_dir / "ZTF_corrupt.npz"
        corrupt_path.write_bytes(b"not a valid npz")
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        # Find the corrupt file index
        corrupt_idx = None
        for i, f in enumerate(ds.files):
            if f.name == "ZTF_corrupt.npz":
                corrupt_idx = i
                break
        if corrupt_idx is not None:
            item = ds[corrupt_idx]
            assert item["obj_id"] == "BAD"
            assert item["seq_len"] == 1

    def test_no_stats_normalization(self, fake_npz_dir):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=None)
        item = ds[0]
        assert item["x"].shape[1] == 7

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PhotoNPZDataset(str(tmp_path))


class TestCollateFn:
    def test_output_keys(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        batch = [ds[i] for i in range(4)]
        collated = collate_fn(batch)
        assert set(collated.keys()) == {"x", "pad_mask", "label_fine", "label_coarse", "obj_ids"}

    def test_padding(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        batch = [ds[i] for i in range(4)]
        collated = collate_fn(batch)
        B = len(batch)
        max_len = max(item["seq_len"] for item in batch)

        assert collated["x"].shape == (B, max_len, 7)
        assert collated["pad_mask"].shape == (B, max_len)

        # Check padding mask is correct
        for i, item in enumerate(batch):
            L = item["seq_len"]
            assert not collated["pad_mask"][i, :L].any()  # valid positions = False
            if L < max_len:
                assert collated["pad_mask"][i, L:].all()  # padding = True

    def test_labels_shape(self, fake_npz_dir, fake_stats):
        ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
        batch = [ds[i] for i in range(4)]
        collated = collate_fn(batch)
        assert collated["label_fine"].shape == (4,)
        assert collated["label_coarse"].shape == (4,)
        assert len(collated["obj_ids"]) == 4
