"""
TransientBind: Cross-Modal Contrastive Learning for Astronomical Transient Alerts.

Learns a unified 128D embedding space where a transient's light curve, cutout
image, and metadata all map to the same point via symmetric InfoNCE loss.

Inspired by: CLIP → ImageBind → AstroCLIP (galaxies) → TransientBind (transients).

Architecture:
    Three independent branch encoders:
        1. LightCurveEncoder: Transformer + CLS pooling → 128D
        2. ImageEncoder: CNN (ImageTower) → 128D
        3. MetadataEncoder: MLP → 128D

    Each encoder outputs a raw (unnormalized) embedding.
    A shared projection head maps embeddings to a contrastive space for training.
    At inference, the raw encoder outputs are used (no projection head).

    Training: Symmetric InfoNCE across all modality pairs (6 directions).
    Inference: Average of L2-normalized encoder embeddings → 128D.

v2 Changelog (post-collapse fix):
    - Added projection heads (encoder output != contrastive space)
    - Removed premature L2 normalization from encoders
    - Added LayerNorm before projection in MetadataEncoder
    - Added learning rate warmup support
    - Fixed augmentation: dropout was zeroing ALL channels (including band one-hots)
    - Fixed augmentation: image flips now vectorized (no Python for-loop)
"""

import torch
import torch.nn.functional as F
from torch import nn

from model import ImageTower, Time2Vec

# ---------------------------------------------------------------------------
# Branch 1: Light Curve Encoder
# ---------------------------------------------------------------------------


class LightCurveEncoder(nn.Module):
    """Encode irregularly sampled multi-band photometry into a 128D embedding.

    Input:  x_phot (B, L, 7) — [log_dt, log_dt_prev, logflux, logflux_err,
                                  band_g, band_r, band_i]
            pad_mask (B, L) — True where padded

    Output: (B, embed_dim) RAW embedding (NOT L2-normalized — normalization
            happens in the contrastive loss, not here)
    """

    def __init__(
        self,
        in_channels: int = 7,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.time2vec = Time2Vec(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_ff,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, x_phot: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        B, _L, _ = x_phot.shape
        h = self.in_proj(x_phot) + self.time2vec(x_phot[..., 0])

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, L+1, d_model)
        pad_ext = F.pad(pad_mask, (1, 0), value=False)  # (B, L+1)

        h = self.encoder(h, src_key_padding_mask=pad_ext)
        cls_out = self.norm(h[:, 0])  # (B, d_model)

        return self.proj(cls_out)  # (B, embed_dim) — NOT normalized


# ---------------------------------------------------------------------------
# Branch 2: Image Encoder
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """Encode ZTF cutout stamps (science, reference, difference) into 128D.

    Input:  images (B, 3, 63, 63)
    Output: (B, embed_dim) RAW embedding (NOT L2-normalized)

    Wraps the existing ImageTower from model.py.
    """

    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 128,
        backbone: str = "simple",
    ):
        super().__init__()
        self.tower = ImageTower(d_model, dropout, backbone=backbone)
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.tower(images)  # (B, d_model)
        return self.proj(h)  # (B, embed_dim) — NOT normalized


# ---------------------------------------------------------------------------
# Branch 3: Metadata Encoder
# ---------------------------------------------------------------------------


class MetadataEncoder(nn.Module):
    """Encode alert metadata fields into 128D.

    Input:  x_meta (B, L, N_META) — per-observation metadata, averaged over time
    Output: (B, embed_dim) RAW embedding (NOT L2-normalized)
    """

    def __init__(
        self,
        in_features: int = 30,
        d_model: int = 256,
        dropout: float = 0.1,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, x_meta: torch.Tensor) -> torch.Tensor:
        # x_meta: (B, L, N_META) — average across time to get static summary
        if x_meta.dim() == 3:
            x_meta = x_meta.mean(dim=1)  # (B, N_META)
        h = self.mlp(x_meta)  # (B, d_model)
        return self.proj(h)  # (B, embed_dim) — NOT normalized


# ---------------------------------------------------------------------------
# Projection Head (the anti-collapse mechanism)
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning.

    This is THE critical component that was missing in v1.
    Every successful contrastive method (SimCLR, CLIP, BYOL, ImageBind)
    uses a projection head between the encoder and the contrastive loss.

    Why it works:
        - The projection head absorbs the destructive contrastive gradients
        - The encoder learns richer, more general representations
        - At inference we DISCARD the projection head and use raw encoder output

    Input:  (B, embed_dim) raw encoder embedding
    Output: (B, proj_dim) L2-normalized projection for contrastive loss
    """

    def __init__(self, embed_dim: int = 128, proj_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Data Augmentation (v2 — fixed)
# ---------------------------------------------------------------------------


def augment_lightcurve(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Apply random augmentations to photometry channels.

    v2 fixes:
        - Dropout only zeros continuous channels (0:4), NOT band one-hots (4:7)
          (zeroing one-hots destroys band identity and confuses the encoder)
        - Flux noise scaled relative to flux_err channel for physical realism
    """
    x_aug = x.clone()
    valid = ~pad_mask  # (B, L)

    # 1. Random observation dropout — zero only continuous channels (0:4)
    drop_prob = torch.rand(x.shape[0], x.shape[1], device=x.device)
    drop_mask = (drop_prob < 0.15) & valid  # 15% dropout (was 20%)
    x_aug[:, :, :4][drop_mask] = 0.0  # only zero dt, dt_prev, flux, flux_err

    # 2. Flux noise — scale by flux_err for physical realism
    flux_err = x[:, :, 3].clamp(min=0.01)  # channel 3 = logflux_err
    noise_scale = flux_err * 0.3  # 30% of reported error
    noise = torch.randn(x.shape[0], x.shape[1], device=x.device) * noise_scale
    x_aug[:, :, 2] = x_aug[:, :, 2] + noise * valid.float()

    # 3. Time jitter — small perturbation to log_dt (channel 0)
    jitter = torch.randn(x.shape[0], x.shape[1], device=x.device) * 0.02
    x_aug[:, :, 0] = x_aug[:, :, 0] + jitter * valid.float()

    return x_aug


def augment_image(images: torch.Tensor) -> torch.Tensor:
    """Apply random augmentations to cutout images.

    v2 fix: vectorized flips (no Python for-loop over batch dimension).
    """
    images = images.clone()
    B = images.shape[0]
    device = images.device

    # Vectorized random horizontal flip
    h_flip = torch.rand(B, device=device) > 0.5
    images[h_flip] = images[h_flip].flip(-1)

    # Vectorized random vertical flip
    v_flip = torch.rand(B, device=device) > 0.5
    images[v_flip] = images[v_flip].flip(-2)

    # Gaussian noise
    images = images + torch.randn_like(images) * 0.02

    return images


# ---------------------------------------------------------------------------
# TransientBind: Full Model (v2)
# ---------------------------------------------------------------------------


class TransientBind(nn.Module):
    """Cross-modal contrastive model for astronomical transient alerts.

    v2 architecture:
        encoder outputs → raw 128D embeddings (for inference)
                        → projection heads → L2-normalized projections (for training loss)

    Training:
        out = model.forward_augmented(x, pad_mask, images)
        loss = model.compute_contrastive_loss(*out)

    Inference:
        embedding = model.embed(x, pad_mask, images)  # (B, 128)
    """

    def __init__(
        self,
        # Light curve encoder
        lc_in_channels: int = 7,
        # Metadata encoder
        meta_in_features: int = 30,
        # Shared
        d_model: int = 256,
        embed_dim: int = 128,
        n_heads: int = 8,
        enc_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        # Image encoder
        image_backbone: str = "simple",
        # Contrastive
        temperature: float = 0.07,
        # Projection head
        proj_dim: int = 128,
        proj_hidden: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Learnable log-temperature (initialized from hyperparameter)
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

        # --- Branch encoders (produce raw embeddings) ---

        self.lc_encoder = LightCurveEncoder(
            in_channels=lc_in_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=enc_layers,
            d_ff=d_ff,
            dropout=dropout,
            embed_dim=embed_dim,
        )

        self.img_encoder = ImageEncoder(
            d_model=d_model,
            dropout=dropout,
            embed_dim=embed_dim,
            backbone=image_backbone,
        )

        self.meta_encoder = MetadataEncoder(
            in_features=meta_in_features,
            d_model=d_model,
            dropout=dropout,
            embed_dim=embed_dim,
        )

        # --- Projection heads (one per modality, used ONLY during training) ---
        self.lc_proj = ProjectionHead(embed_dim, proj_dim, proj_hidden)
        self.img_proj = ProjectionHead(embed_dim, proj_dim, proj_hidden)
        self.meta_proj = ProjectionHead(embed_dim, proj_dim, proj_hidden)

    def _encode(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw encoder outputs (NOT projected, NOT normalized).

        These are the representations used at inference time.
        """
        x_phot = x[:, :, :7]
        x_meta = x[:, :, 7:]

        lc_emb = self.lc_encoder(x_phot, pad_mask)  # (B, embed_dim)
        img_emb = self.img_encoder(images)  # (B, embed_dim)
        meta_emb = self.meta_encoder(x_meta)  # (B, embed_dim)

        return lc_emb, img_emb, meta_emb

    def _project(
        self,
        lc_emb: torch.Tensor,
        img_emb: torch.Tensor,
        meta_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project raw embeddings into contrastive space (L2-normalized)."""
        return (
            self.lc_proj(lc_emb),
            self.img_proj(img_emb),
            self.meta_proj(meta_emb),
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and project all three modalities (for training).

        Returns PROJECTED, L2-normalized embeddings for contrastive loss.
        """
        lc_emb, img_emb, meta_emb = self._encode(x, pad_mask, images)
        return self._project(lc_emb, img_emb, meta_emb)

    def forward_augmented(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with data augmentation (use during training).

        Returns PROJECTED, L2-normalized embeddings for contrastive loss.
        """
        x_aug = x.clone()
        x_aug[:, :, :7] = augment_lightcurve(x[:, :, :7], pad_mask)
        images_aug = augment_image(images)

        x_phot = x_aug[:, :, :7]
        x_meta = x[:, :, 7:]  # metadata not augmented

        lc_emb = self.lc_encoder(x_phot, pad_mask)
        img_emb = self.img_encoder(images_aug)
        meta_emb = self.meta_encoder(x_meta)

        return self._project(lc_emb, img_emb, meta_emb)

    def compute_contrastive_loss(
        self,
        lc_proj: torch.Tensor,
        img_proj: torch.Tensor,
        meta_proj: torch.Tensor,
    ) -> dict:
        """Symmetric InfoNCE loss across all modality pairs.

        Args:
            lc_proj:   (B, proj_dim) L2-normalized projections
            img_proj:  (B, proj_dim) L2-normalized projections
            meta_proj: (B, proj_dim) L2-normalized projections

        Returns:
            dict with 'total_loss', 'loss_lc_img', 'loss_lc_meta',
            'loss_img_meta', 'acc_lc_img', 'acc_lc_meta', 'acc_img_meta'
        """
        B = lc_proj.shape[0]
        device = lc_proj.device
        labels = torch.arange(B, device=device)

        # Effective temperature (clamped for stability)
        temp = self.log_temperature.exp().clamp(min=0.01, max=1.0)

        # LC ↔ Image
        logits_lc_img = (lc_proj @ img_proj.T) / temp  # (B, B)
        loss_lc2img = F.cross_entropy(logits_lc_img, labels)
        loss_img2lc = F.cross_entropy(logits_lc_img.T, labels)
        loss_lc_img = (loss_lc2img + loss_img2lc) / 2

        # LC ↔ Metadata
        logits_lc_meta = (lc_proj @ meta_proj.T) / temp
        loss_lc2meta = F.cross_entropy(logits_lc_meta, labels)
        loss_meta2lc = F.cross_entropy(logits_lc_meta.T, labels)
        loss_lc_meta = (loss_lc2meta + loss_meta2lc) / 2

        # Image ↔ Metadata
        logits_img_meta = (img_proj @ meta_proj.T) / temp
        loss_img2meta = F.cross_entropy(logits_img_meta, labels)
        loss_meta2img = F.cross_entropy(logits_img_meta.T, labels)
        loss_img_meta = (loss_img2meta + loss_meta2img) / 2

        total_loss = (loss_lc_img + loss_lc_meta + loss_img_meta) / 3

        # Retrieval accuracy (for monitoring, not training)
        with torch.no_grad():
            acc_lc_img = (logits_lc_img.argmax(dim=1) == labels).float().mean().item()
            acc_lc_meta = (logits_lc_meta.argmax(dim=1) == labels).float().mean().item()
            acc_img_meta = (logits_img_meta.argmax(dim=1) == labels).float().mean().item()

        return {
            "total_loss": total_loss,
            "loss_lc_img": loss_lc_img.item(),
            "loss_lc_meta": loss_lc_meta.item(),
            "loss_img_meta": loss_img_meta.item(),
            "acc_lc_img": acc_lc_img,
            "acc_lc_meta": acc_lc_meta,
            "acc_img_meta": acc_img_meta,
            "temperature": temp.item(),
        }

    @torch.no_grad()
    def embed(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Produce fused 128D embedding for downstream use.

        Uses RAW encoder outputs (no projection head). Averages L2-normalized
        branch embeddings and re-normalizes.

        Compatible with RTF's embed() interface.

        Returns:
            (B, 128) L2-normalized embedding
        """
        lc_emb, img_emb, meta_emb = self._encode(x, pad_mask, images)
        # Normalize each branch independently, then average
        lc_n = F.normalize(lc_emb, dim=-1)
        img_n = F.normalize(img_emb, dim=-1)
        meta_n = F.normalize(meta_emb, dim=-1)
        fused = (lc_n + img_n + meta_n) / 3
        return F.normalize(fused, dim=-1)

    @torch.no_grad()
    def embed_per_modality(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-modality embeddings (for cross-modal retrieval eval).

        Returns:
            dict with 'lc' (B, 128), 'img' (B, 128), 'meta' (B, 128), 'fused' (B, 128)
        """
        lc_emb, img_emb, meta_emb = self._encode(x, pad_mask, images)
        lc_n = F.normalize(lc_emb, dim=-1)
        img_n = F.normalize(img_emb, dim=-1)
        meta_n = F.normalize(meta_emb, dim=-1)
        fused = F.normalize((lc_n + img_n + meta_n) / 3, dim=-1)
        return {
            "lc": lc_n,
            "img": img_n,
            "meta": meta_n,
            "fused": fused,
        }

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Anomaly score based on cross-modal disagreement.

        TransientBind is contrastive (no decoder), so this returns the negative
        of the mean cosine similarity between branch embeddings as a proxy
        for how well the modalities agree. Higher = more disagreement = more unusual.

        Returns:
            (B,) anomaly-like score per sample
        """
        lc_emb, img_emb, meta_emb = self._encode(x, pad_mask, images)
        lc_n = F.normalize(lc_emb, dim=-1)
        img_n = F.normalize(img_emb, dim=-1)
        meta_n = F.normalize(meta_emb, dim=-1)
        sim_lc_img = (lc_n * img_n).sum(dim=-1)
        sim_lc_meta = (lc_n * meta_n).sum(dim=-1)
        sim_img_meta = (img_n * meta_n).sum(dim=-1)
        agreement = (sim_lc_img + sim_lc_meta + sim_img_meta) / 3
        return -agreement

    def compression_info(self) -> dict:
        """Compression stats for logging (matches RTF interface)."""
        return {
            "latent_dim": self.embed_dim,
            "compressed_bytes": self.embed_dim * 4,
            "compression_ratio": 0.0,
        }
