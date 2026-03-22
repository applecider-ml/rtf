"""Tests for LightCurveCompressor model (AE, VAE, VQ-VAE)."""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import LightCurveCompressor, Time2Vec, VectorQuantizer


# ---------------------------------------------------------------------------
# Time2Vec
# ---------------------------------------------------------------------------


class TestTime2Vec:
    def test_output_shape(self):
        t2v = Time2Vec(64)
        t = torch.randn(2, 10)
        out = t2v(t)
        assert out.shape == (2, 10, 64)

    def test_differentiable(self):
        t2v = Time2Vec(32)
        t = torch.randn(2, 5, requires_grad=True)
        out = t2v(t)
        out.sum().backward()
        assert t.grad is not None


# ---------------------------------------------------------------------------
# VectorQuantizer
# ---------------------------------------------------------------------------


class TestVectorQuantizer:
    def test_output_shape(self):
        vq = VectorQuantizer(num_codes=64, code_dim=16)
        z_e = torch.randn(8, 16)
        z_q, loss, indices = vq(z_e)
        assert z_q.shape == (8, 16)
        assert loss.shape == ()
        assert indices.shape == (8,)

    def test_indices_in_range(self):
        vq = VectorQuantizer(num_codes=64, code_dim=16)
        z_e = torch.randn(32, 16)
        _, _, indices = vq(z_e)
        assert (indices >= 0).all() and (indices < 64).all()

    def test_straight_through_gradient(self):
        vq = VectorQuantizer(num_codes=64, code_dim=16)
        z_e = torch.randn(4, 16, requires_grad=True)
        z_q, loss, _ = vq(z_e)
        (z_q.sum() + loss).backward()
        assert z_e.grad is not None


# ---------------------------------------------------------------------------
# LightCurveCompressor — shared tests across modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ae", "vae", "vqvae"])
class TestCompressorAllModes:
    def test_forward_shapes(self, mode, sample_batch):
        model = LightCurveCompressor(
            mode=mode, latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        x = sample_batch["x"]
        pad = sample_batch["pad_mask"]
        out = model(x, pad)

        B, L = x.shape[:2]
        if mode == "ae":
            cont_hat, band_logits, z = out
            assert z.shape == (B, 16)
        elif mode == "vae":
            cont_hat, band_logits, mu, logvar = out
            assert mu.shape == (B, 16)
            assert logvar.shape == (B, 16)
        elif mode == "vqvae":
            cont_hat, band_logits, vq_loss, indices = out
            assert vq_loss.shape == ()
            assert indices.shape == (B,)

        assert cont_hat.shape == (B, L, 4)
        assert band_logits.shape == (B, L, 3)

    def test_loss_computation(self, mode, sample_batch):
        model = LightCurveCompressor(
            mode=mode, latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        x = sample_batch["x"]
        pad = sample_batch["pad_mask"]
        out = model(x, pad)
        loss_dict = model.compute_loss(x, pad, out, beta=0.5)

        assert "total_loss" in loss_dict
        assert "recon_cont" in loss_dict
        assert "recon_band" in loss_dict
        assert "band_acc" in loss_dict
        assert loss_dict["total_loss"].requires_grad
        assert loss_dict["recon_cont"] >= 0
        assert loss_dict["recon_band"] >= 0
        assert 0 <= loss_dict["band_acc"] <= 1

    def test_embed(self, mode, sample_batch):
        model = LightCurveCompressor(
            mode=mode, latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        x = sample_batch["x"]
        pad = sample_batch["pad_mask"]
        emb = model.embed(x, pad)
        assert emb.shape == (x.shape[0], 16)
        assert not emb.requires_grad

    def test_reconstruction_error(self, mode, sample_batch):
        model = LightCurveCompressor(
            mode=mode, latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        x = sample_batch["x"]
        pad = sample_batch["pad_mask"]
        err = model.reconstruction_error(x, pad)
        assert err.shape == (x.shape[0],)
        assert (err >= 0).all()

    def test_backward_pass(self, mode, sample_batch):
        model = LightCurveCompressor(
            mode=mode, latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        x = sample_batch["x"]
        pad = sample_batch["pad_mask"]
        out = model(x, pad)
        loss_dict = model.compute_loss(x, pad, out, beta=0.5)
        loss_dict["total_loss"].backward()
        # Check gradients flow to encoder
        assert model.in_proj.weight.grad is not None
        assert model.in_proj.weight.grad.abs().sum() > 0

    def test_compression_info(self, mode):
        model = LightCurveCompressor(mode=mode, latent_dim=64)
        info = model.compression_info()
        assert info["mode"] == mode
        assert info["latent_dim"] == 64
        assert info["compression_ratio"] > 0
        assert info["compressed_bytes"] > 0


# ---------------------------------------------------------------------------
# Mode-specific tests
# ---------------------------------------------------------------------------


class TestAE:
    def test_deterministic_output(self, sample_batch):
        model = LightCurveCompressor(
            mode="ae", latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        model.eval()
        x = sample_batch["x"]
        pad = sample_batch["pad_mask"]
        out1 = model(x, pad)
        out2 = model(x, pad)
        assert torch.allclose(out1[0], out2[0])  # cont_hat identical

    def test_no_kl_in_loss(self, sample_batch):
        model = LightCurveCompressor(
            mode="ae", latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        x, pad = sample_batch["x"], sample_batch["pad_mask"]
        loss_dict = model.compute_loss(x, pad, model(x, pad), beta=1.0)
        assert loss_dict["kld"] == 0.0


class TestVAE:
    def test_stochastic_during_training(self, sample_batch):
        model = LightCurveCompressor(
            mode="vae", latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        model.train()
        x = sample_batch["x"]
        pad = sample_batch["pad_mask"]
        out1 = model(x, pad)
        out2 = model(x, pad)
        # Outputs should differ due to reparameterization noise
        assert not torch.allclose(out1[0], out2[0])

    def test_deterministic_during_eval(self, sample_batch):
        model = LightCurveCompressor(
            mode="vae", latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        model.eval()
        x = sample_batch["x"]
        pad = sample_batch["pad_mask"]
        out1 = model(x, pad)
        out2 = model(x, pad)
        # eval uses mu directly, no noise
        assert torch.allclose(out1[0], out2[0])

    def test_kl_positive(self, sample_batch):
        model = LightCurveCompressor(
            mode="vae", latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        x, pad = sample_batch["x"], sample_batch["pad_mask"]
        loss_dict = model.compute_loss(x, pad, model(x, pad), beta=1.0)
        assert loss_dict["kld"] > 0

    def test_beta_scaling(self, sample_batch):
        model = LightCurveCompressor(
            mode="vae", latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
        )
        x, pad = sample_batch["x"], sample_batch["pad_mask"]
        out = model(x, pad)
        loss_b0 = model.compute_loss(x, pad, out, beta=0.0)
        loss_b1 = model.compute_loss(x, pad, out, beta=1.0)
        # With beta=0, total = recon only; with beta=1, total = recon + kl
        assert loss_b1["total_loss"].item() > loss_b0["total_loss"].item()


class TestVQVAE:
    def test_codebook_usage(self, sample_batch):
        model = LightCurveCompressor(
            mode="vqvae",
            latent_dim=16,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            num_codes=64,
        )
        x, pad = sample_batch["x"], sample_batch["pad_mask"]
        loss_dict = model.compute_loss(x, pad, model(x, pad))
        assert loss_dict["codebook_usage"] > 0
        assert loss_dict["codebook_usage"] <= 64

    def test_vq_loss_positive(self, sample_batch):
        model = LightCurveCompressor(
            mode="vqvae",
            latent_dim=16,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
        )
        x, pad = sample_batch["x"], sample_batch["pad_mask"]
        loss_dict = model.compute_loss(x, pad, model(x, pad))
        assert loss_dict["vq_loss"] > 0


# ---------------------------------------------------------------------------
# Latent dimension variations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent_dim", [2, 8, 32, 128, 256])
def test_various_latent_dims(sample_batch, latent_dim):
    model = LightCurveCompressor(
        mode="ae",
        latent_dim=latent_dim,
        d_model=32,
        n_heads=4,
        enc_layers=1,
        dec_layers=1,
    )
    x, pad = sample_batch["x"], sample_batch["pad_mask"]
    out = model(x, pad)
    loss_dict = model.compute_loss(x, pad, out)
    assert loss_dict["total_loss"].isfinite()
    emb = model.embed(x, pad)
    assert emb.shape == (x.shape[0], latent_dim)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(sample_batch, tmp_path):
    model = LightCurveCompressor(
        mode="vae", latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
    )
    model.eval()
    x, pad = sample_batch["x"], sample_batch["pad_mask"]
    emb_before = model.embed(x, pad)

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    model2 = LightCurveCompressor(
        mode="vae", latent_dim=16, d_model=32, n_heads=4, enc_layers=1, dec_layers=1
    )
    model2.load_state_dict(torch.load(path, weights_only=True))
    model2.eval()
    emb_after = model2.embed(x, pad)

    assert torch.allclose(emb_before, emb_after)


# ---------------------------------------------------------------------------
# Image tower
# ---------------------------------------------------------------------------


class TestImageTower:
    def test_output_shape(self):
        from model import ImageTower

        tower = ImageTower(d_model=32)
        imgs = torch.randn(8, 3, 63, 63)
        out = tower(imgs)
        assert out.shape == (8, 32)

    def test_gradient_flow(self):
        from model import ImageTower

        tower = ImageTower(d_model=32)
        imgs = torch.randn(4, 3, 63, 63, requires_grad=True)
        out = tower(imgs)
        out.sum().backward()
        assert imgs.grad is not None


# ---------------------------------------------------------------------------
# Multimodal: images
# ---------------------------------------------------------------------------


class TestWithImages:
    def test_forward(self, sample_batch_images):
        model = LightCurveCompressor(
            mode="ae",
            latent_dim=16,
            in_channels=37,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            use_images=True,
        )
        b = sample_batch_images
        out = model(b["x"], b["pad_mask"], b["images"])
        assert out[0].shape == (4, 15, 4)
        assert out[2].shape == (4, 16)

    def test_embed(self, sample_batch_images):
        model = LightCurveCompressor(
            mode="ae",
            latent_dim=16,
            in_channels=37,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            use_images=True,
        )
        b = sample_batch_images
        emb = model.embed(b["x"], b["pad_mask"], b["images"])
        assert emb.shape == (4, 16)

    def test_backward(self, sample_batch_images):
        model = LightCurveCompressor(
            mode="ae",
            latent_dim=16,
            in_channels=37,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            use_images=True,
        )
        b = sample_batch_images
        out = model(b["x"], b["pad_mask"], b["images"])
        loss = model.compute_loss(b["x"], b["pad_mask"], out)
        loss["total_loss"].backward()
        assert model.image_tower.cnn[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Multimodal: GP features
# ---------------------------------------------------------------------------


class TestWithGP:
    def test_forward(self, sample_batch_gp):
        model = LightCurveCompressor(
            mode="ae",
            latent_dim=16,
            in_channels=37,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            gp_dim=12,
        )
        b = sample_batch_gp
        out = model(b["x"], b["pad_mask"], gp_features=b["gp_features"])
        assert out[0].shape == (4, 15, 4)
        assert out[2].shape == (4, 16)

    def test_embed(self, sample_batch_gp):
        model = LightCurveCompressor(
            mode="ae",
            latent_dim=16,
            in_channels=37,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            gp_dim=12,
        )
        b = sample_batch_gp
        emb = model.embed(b["x"], b["pad_mask"], gp_features=b["gp_features"])
        assert emb.shape == (4, 16)

    def test_backward(self, sample_batch_gp):
        model = LightCurveCompressor(
            mode="ae",
            latent_dim=16,
            in_channels=37,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            gp_dim=12,
        )
        b = sample_batch_gp
        out = model(b["x"], b["pad_mask"], gp_features=b["gp_features"])
        loss = model.compute_loss(b["x"], b["pad_mask"], out)
        loss["total_loss"].backward()
        assert model.gp_proj[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Full multimodal: images + GP
# ---------------------------------------------------------------------------


class TestAllModalities:
    def test_forward(self, sample_batch_all):
        model = LightCurveCompressor(
            mode="ae",
            latent_dim=16,
            in_channels=37,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            use_images=True,
            gp_dim=12,
        )
        b = sample_batch_all
        out = model(b["x"], b["pad_mask"], b["images"], gp_features=b["gp_features"])
        assert out[0].shape == (4, 15, 4)

    def test_backward(self, sample_batch_all):
        model = LightCurveCompressor(
            mode="ae",
            latent_dim=16,
            in_channels=37,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            use_images=True,
            gp_dim=12,
        )
        b = sample_batch_all
        out = model(b["x"], b["pad_mask"], b["images"], gp_features=b["gp_features"])
        loss = model.compute_loss(b["x"], b["pad_mask"], out)
        loss["total_loss"].backward()
        assert model.image_tower.cnn[0].weight.grad is not None
        assert model.gp_proj[0].weight.grad is not None
        assert model.in_proj.weight.grad is not None
