"""Integration tests: training loop, embedding extraction, linear probe."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import PhotoNPZDataset, collate_fn
from model import LightCurveCompressor


@pytest.fixture
def small_loader(fake_npz_dir, fake_stats):
    ds = PhotoNPZDataset(str(fake_npz_dir), stats_path=fake_stats)
    return DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)


@pytest.mark.parametrize("mode", ["ae", "vae", "vqvae"])
class TestTrainingLoop:
    def test_loss_decreases(self, mode, small_loader):
        """Train for a few steps and verify loss moves."""
        model = LightCurveCompressor(
            mode=mode, latent_dim=8, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        model.train()
        for epoch in range(5):
            epoch_loss = 0
            n = 0
            for batch in small_loader:
                x, pad = batch["x"], batch["pad_mask"]
                out = model(x, pad)
                loss_dict = model.compute_loss(x, pad, out, beta=0.1)
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                optimizer.step()
                epoch_loss += loss_dict["total_loss"].item()
                n += 1
            losses.append(epoch_loss / n)

        # Loss should not be NaN/Inf
        assert all(np.isfinite(v) for v in losses)
        # Loss should decrease or at least not explode
        assert losses[-1] < losses[0] * 2

    def test_embedding_extraction(self, mode, small_loader):
        model = LightCurveCompressor(
            mode=mode, latent_dim=8, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        model.eval()
        all_emb = []
        with torch.no_grad():
            for batch in small_loader:
                emb = model.embed(batch["x"], batch["pad_mask"])
                all_emb.append(emb)
        emb = torch.cat(all_emb)
        assert emb.shape[1] == 8  # latent_dim=8
        assert emb.shape[0] == len(small_loader.dataset)
        assert emb.isfinite().all()

    def test_recon_error_extraction(self, mode, small_loader):
        model = LightCurveCompressor(
            mode=mode, latent_dim=8, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        model.eval()
        all_err = []
        with torch.no_grad():
            for batch in small_loader:
                err = model.reconstruction_error(batch["x"], batch["pad_mask"])
                all_err.append(err)
        err = torch.cat(all_err)
        assert err.shape == (len(small_loader.dataset),)
        assert (err >= 0).all()
        assert err.isfinite().all()


def test_gradient_clipping(small_loader):
    """Verify gradient clipping doesn't break training."""
    model = LightCurveCompressor(
        mode="vae", latent_dim=8, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    batch = next(iter(small_loader))
    out = model(batch["x"], batch["pad_mask"])
    loss = model.compute_loss(batch["x"], batch["pad_mask"], out, beta=1.0)
    optimizer.zero_grad()
    loss["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # All grads should be finite after clipping
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.isfinite().all()
    optimizer.step()


def test_save_load_produces_same_embeddings(small_loader, tmp_path):
    """Save/load roundtrip preserves inference behavior."""
    model = LightCurveCompressor(
        mode="ae", latent_dim=8, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
    )
    model.eval()
    batch = next(iter(small_loader))
    emb1 = model.embed(batch["x"], batch["pad_mask"])

    path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), path)

    model2 = LightCurveCompressor(
        mode="ae", latent_dim=8, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
    )
    model2.load_state_dict(torch.load(path, weights_only=True))
    model2.eval()
    emb2 = model2.embed(batch["x"], batch["pad_mask"])

    assert torch.allclose(emb1, emb2, atol=1e-6)
