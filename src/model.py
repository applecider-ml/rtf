"""
Light curve compression models: AE, VAE, and VQ-VAE variants.

All three share the same transformer encoder/decoder backbone and differ only
in the bottleneck:
  - AE:     deterministic bottleneck, no regularization
  - VAE:    continuous Gaussian latent with KL divergence
  - VQ-VAE: discrete codebook with straight-through estimator

Input: (B, L, 7) padded photometry sequences
  channels: [log1p(dt), log1p(dt_prev), logflux, logflux_err, band_g, band_r, band_i]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """Learned time embedding: linear + periodic components."""

    def __init__(self, d_model: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        v0 = self.w0 * t + self.b0
        vp = torch.sin(t.unsqueeze(-1) * self.w + self.b)
        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)


class VectorQuantizer(nn.Module):
    """Vector quantization layer for VQ-VAE.

    Maps continuous encoder output to nearest codebook entry.
    Uses exponential moving average (EMA) codebook updates for stability.
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        # EMA state
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed_sum", self.codebook.weight.clone())

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: (B, code_dim) continuous encoder output

        Returns:
            z_q: (B, code_dim) quantized vector (straight-through)
            vq_loss: commitment + codebook loss
            indices: (B,) codebook indices
        """
        # Distances to codebook entries
        dists = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )
        indices = dists.argmin(dim=1)
        z_q = self.codebook(indices)

        if self.training:
            # EMA codebook update
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    one_hot.sum(0), alpha=1 - self.ema_decay
                )
                embed_sum = one_hot.t() @ z_e
                self.ema_embed_sum.mul_(self.ema_decay).add_(
                    embed_sum, alpha=1 - self.ema_decay
                )
                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
                )
                self.codebook.weight.data.copy_(
                    self.ema_embed_sum / cluster_size.unsqueeze(1)
                )

        # Losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)  # commitment
        vq_loss = self.commitment_cost * codebook_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, vq_loss, indices


class ImageTower(nn.Module):
    """CNN for 63x63 cutout stamps → d_model embedding.

    Supports two backends:
      - "simple": lightweight 4-layer CNN (~260K params), trained from scratch
      - "zoobot": Galaxy Zoo pretrained ConvNeXt-pico via timm (~8.5M params)

    Processes (science, template, difference) 3-channel stamps.
    Applied per-observation in the sequence, so input is (B*L, 3, 63, 63).
    """

    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.2,
        backbone: str = "simple",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone

        if backbone == "zoobot":
            self._init_zoobot(d_model, dropout, freeze_backbone)
        else:
            self._init_simple(d_model, dropout)

    def _init_simple(self, d_model, dropout):
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )

    def _init_zoobot(self, d_model, dropout, freeze_backbone):
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for zoobot backbone. Install with: pip install timm"
            )

        # Load ConvNeXt-pico pretrained on Galaxy Zoo via Zoobot
        # Falls back to ImageNet pretrained if Zoobot weights not available
        try:
            self.cnn = timm.create_model(
                "hf_hub:mwalmsley/zoobot-encoder-convnext_pico",
                pretrained=True,
                num_classes=0,  # remove classification head, get features
            )
        except Exception:
            print(
                "  Warning: Zoobot weights not available, using ImageNet ConvNeXt-pico"
            )
            self.cnn = timm.create_model(
                "convnext_pico", pretrained=True, num_classes=0
            )

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # The backbone outputs a feature vector; project to d_model
        feat_dim = self.cnn.num_features
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, d_model),
        )

    def _run_cnn_subbatch(
        self, images: torch.Tensor, max_sub: int = 32
    ) -> torch.Tensor:
        """Run CNN in sub-batches, with or without gradients."""
        if self.freeze_backbone:
            # No gradients needed for frozen backbone
            ctx = torch.no_grad
        else:
            # Use gradient checkpointing to save memory during fine-tuning
            from contextlib import nullcontext

            ctx = nullcontext

        feats = []
        for i in range(0, images.shape[0], max_sub):
            chunk = images[i : i + max_sub]
            with ctx():
                if not self.freeze_backbone and hasattr(torch.utils, "checkpoint"):
                    feat = torch.utils.checkpoint.checkpoint(
                        self.cnn, chunk, use_reentrant=False
                    )
                else:
                    feat = self.cnn(chunk)
            feats.append(feat)
        return torch.cat(feats, dim=0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (N, 3, 63, 63) → (N, d_model)"""
        if self.backbone_name == "zoobot":
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )
            cnn_out = self._run_cnn_subbatch(images)
            return self.proj(cnn_out)
        return self.proj(self.cnn(images))


class LightCurveCompressor(nn.Module):
    """
    Unified transformer autoencoder for light curve compression.

    Modes:
        "ae"    — deterministic autoencoder
        "vae"   — variational autoencoder with KL regularization
        "vqvae" — vector-quantized VAE with discrete codebook

    If use_images=True, a CNN image tower processes per-observation cutout stamps
    and adds image embeddings to the sequence representation before the transformer.
    """

    def __init__(
        self,
        mode: str = "vae",
        latent_dim: int = 32,
        in_channels: int = 7,
        d_model: int = 128,
        n_heads: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.3,
        max_len: int = 257,
        # VAE-specific
        free_bits: float = 0.25,
        # VQ-VAE-specific
        num_codes: int = 512,
        commitment_cost: float = 0.25,
        # Image tower
        use_images: bool = False,
        image_backbone: str = "simple",
        freeze_backbone: bool = None,  # None = auto (freeze for zoobot, not for simple)
        # GP features
        gp_dim: int = 0,
        # Classification head (joint training)
        num_classes: int = 0,
    ):
        super().__init__()
        assert mode in ("ae", "vae", "vqvae"), f"Unknown mode: {mode}"
        self.mode = mode
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.d_model = d_model
        self.max_len = max_len
        self.free_bits = free_bits
        self.use_images = use_images
        self.gp_dim = gp_dim
        self.num_classes = num_classes

        # --- Image tower (optional) ---
        if use_images:
            freeze = (
                freeze_backbone
                if freeze_backbone is not None
                else (image_backbone == "zoobot")
            )
            self.image_tower = ImageTower(
                d_model, dropout, backbone=image_backbone, freeze_backbone=freeze
            )

        # --- GP feature projection (optional) ---
        # Projects static per-source GP features and adds to CLS token
        if gp_dim > 0:
            self.gp_proj = nn.Sequential(
                nn.Linear(gp_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # --- Encoder ---
        self.in_proj = nn.Linear(in_channels, d_model)
        self.time2vec = Time2Vec(d_model)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_tok, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_ff,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        self.enc_norm = nn.LayerNorm(d_model)

        # --- Bottleneck (mode-dependent) ---
        if mode == "vae":
            self.mu_proj = nn.Linear(d_model, latent_dim)
            self.logvar_proj = nn.Linear(d_model, latent_dim)
        elif mode == "ae":
            self.bottleneck_proj = nn.Linear(d_model, latent_dim)
        elif mode == "vqvae":
            self.pre_vq_proj = nn.Linear(d_model, latent_dim)
            self.vq = VectorQuantizer(num_codes, latent_dim, commitment_cost)

        # --- Classification head (optional, for joint training) ---
        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, num_classes),
            )

        # --- Decoder ---
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        dec_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_ff,
            max(dropout - 0.1, 0.1),
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, dec_layers)
        self.dec_norm = nn.LayerNorm(d_model)

        self.head_cont = nn.Linear(d_model, 4)
        self.head_band = nn.Linear(d_model, 3)

    def _encode_cls(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor = None,
        gp_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run encoder, return CLS token representation.

        Images: (B, 3, 63, 63) — single most-recent stamp per source.
        GP features: (B, gp_dim) — static per-source vector.
        Both are fused additively with the CLS token after the transformer.
        """
        B, L, _ = x.shape
        h = self.in_proj(x) + self.time2vec(x[..., 0])

        cls = self.cls_tok.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        pad_ext = F.pad(pad_mask, (1, 0), value=False)
        z = self.encoder(h, src_key_padding_mask=pad_ext)
        cls_out = self.enc_norm(z[:, 0])

        # Fuse image embedding: most recent stamp only (matches alert packet)
        # Process B images (not B*L), add to CLS token like GP features
        if self.use_images and images is not None:
            # images: (B, 3, 63, 63) — single stamp per source
            img_emb = self.image_tower(images)  # (B, d_model)
            cls_out = cls_out + img_emb

        # Fuse GP features: additive to CLS token
        if self.gp_dim > 0 and gp_features is not None:
            gp_emb = self.gp_proj(gp_features)
            cls_out = cls_out + gp_emb

        return cls_out

    def encode(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor = None,
        gp_features: torch.Tensor = None,
    ):
        """
        Encode to latent representation.

        Returns depend on mode:
            ae:    (z,)
            vae:   (mu, logvar)
            vqvae: (z_q, vq_loss, indices)
        """
        cls_out = self._encode_cls(x, pad_mask, images, gp_features)

        if self.mode == "ae":
            z = self.bottleneck_proj(cls_out)
            return (z,)
        elif self.mode == "vae":
            mu = self.mu_proj(cls_out)
            logvar = self.logvar_proj(cls_out)
            return (mu, logvar)
        elif self.mode == "vqvae":
            z_e = self.pre_vq_proj(cls_out)
            z_q, vq_loss, indices = self.vq(z_e)
            return (z_q, vq_loss, indices)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> tuple:
        """Decode latent vector to (cont_hat, band_logits)."""
        h = self.z_proj(z).unsqueeze(1).expand(-1, self.max_len, -1)
        h = h + self.pos_embed[:, : self.max_len, :]
        h = self.dec_norm(self.decoder(h))
        return self.head_cont(h), self.head_band(h)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor = None,
        gp_features: torch.Tensor = None,
    ):
        """
        Full forward pass.

        Returns:
            cont_hat, band_logits, and mode-specific extras:
              ae:    cont_hat, band_logits, z
              vae:   cont_hat, band_logits, mu, logvar
              vqvae: cont_hat, band_logits, vq_loss, indices
        """
        L = x.shape[1]

        if self.mode == "ae":
            (z,) = self.encode(x, pad_mask, images, gp_features)
            cont_hat, band_logits = self.decode(z)
            return cont_hat[:, :L], band_logits[:, :L], z

        elif self.mode == "vae":
            mu, logvar = self.encode(x, pad_mask, images, gp_features)
            z = self.reparameterize(mu, logvar)
            cont_hat, band_logits = self.decode(z)
            return cont_hat[:, :L], band_logits[:, :L], mu, logvar

        elif self.mode == "vqvae":
            z_q, vq_loss, indices = self.encode(x, pad_mask, images, gp_features)
            cont_hat, band_logits = self.decode(z_q)
            return cont_hat[:, :L], band_logits[:, :L], vq_loss, indices

    def compute_loss(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        forward_out: tuple,
        beta: float = 1.0,
        labels: torch.Tensor = None,
        cls_weight: float = 0.0,
    ) -> dict:
        """
        Compute loss for any mode.

        Args:
            x: (B, L, C) input
            pad_mask: (B, L) padding mask
            forward_out: output tuple from forward()
            beta: KL weight (VAE only)
            labels: (B,) class labels for joint classification (optional)
            cls_weight: weight for classification loss (0 = reconstruction only)
        """
        valid = ~pad_mask
        n_valid = valid.sum().clamp(min=1)
        B, L = pad_mask.shape

        if self.mode == "ae":
            cont_hat, band_logits, z = forward_out
        elif self.mode == "vae":
            cont_hat, band_logits, mu, logvar = forward_out
        elif self.mode == "vqvae":
            cont_hat, band_logits, vq_loss, indices = forward_out

        # Continuous reconstruction
        cont_target = x[..., :4]
        mse = ((cont_hat - cont_target) ** 2) * valid.unsqueeze(-1)
        recon_cont = mse.sum() / (n_valid * 4)

        # Per-channel MSE
        channel_names = ["dt", "dt_prev", "logflux", "logflux_err"]
        per_ch = {}
        for i, name in enumerate(channel_names):
            per_ch[f"mse_{name}"] = (
                (cont_hat[..., i] - cont_target[..., i]) ** 2 * valid
            ).sum().item() / n_valid.item()

        # Band reconstruction
        band_target = x[..., 4:7].argmax(-1)
        ce = F.cross_entropy(
            band_logits.reshape(-1, 3), band_target.reshape(-1), reduction="none"
        )
        recon_band = (ce.reshape(B, L) * valid).sum() / n_valid

        # Band accuracy
        band_acc = (
            (band_logits.argmax(-1) == band_target) * valid
        ).sum().item() / n_valid.item()

        # Extract latent vector for classification (same for all modes)
        if self.mode == "ae":
            z_cls = z
        elif self.mode == "vae":
            z_cls = mu
        elif self.mode == "vqvae":
            # vq_loss and indices are in forward_out[2:], z_q was used for decoding
            # Recompute from forward_out — the quantized vector is what was decoded from
            z_cls = None  # VQ-VAE cls not supported yet

        # Classification loss (optional joint training)
        cls_loss_val = 0.0
        cls_acc = 0.0
        cls_loss_tensor = torch.tensor(0.0, device=x.device)
        if (
            self.num_classes > 0
            and labels is not None
            and cls_weight > 0
            and z_cls is not None
        ):
            cls_logits = self.cls_head(z_cls)
            cls_loss_tensor = F.cross_entropy(cls_logits, labels)
            cls_loss_val = cls_loss_tensor.item()
            cls_acc = (cls_logits.argmax(-1) == labels).float().mean().item()

        # Mode-specific regularization
        if self.mode == "ae":
            total = recon_cont + recon_band + cls_weight * cls_loss_tensor
            return {
                "total_loss": total,
                "recon_cont": recon_cont.item(),
                "recon_band": recon_band.item(),
                "band_acc": band_acc,
                "cls_loss": cls_loss_val,
                "cls_acc": cls_acc,
                "beta": 0.0,
                "kld": 0.0,
                "vq_loss": 0.0,
                **per_ch,
            }

        elif self.mode == "vae":
            kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            if self.free_bits > 0:
                kld_per_dim = kld_per_dim.clamp(min=self.free_bits)
            kld = kld_per_dim.sum(-1).mean()
            total = recon_cont + recon_band + beta * kld + cls_weight * cls_loss_tensor
            return {
                "total_loss": total,
                "recon_cont": recon_cont.item(),
                "recon_band": recon_band.item(),
                "kld": kld.item(),
                "beta": beta,
                "band_acc": band_acc,
                "cls_loss": cls_loss_val,
                "cls_acc": cls_acc,
                "vq_loss": 0.0,
                **per_ch,
            }

        elif self.mode == "vqvae":
            total = recon_cont + recon_band + vq_loss + cls_weight * cls_loss_tensor
            # Codebook utilization
            n_unique = indices.unique().numel()
            return {
                "total_loss": total,
                "recon_cont": recon_cont.item(),
                "recon_band": recon_band.item(),
                "vq_loss": vq_loss.item(),
                "codebook_usage": n_unique,
                "beta": 0.0,
                "kld": 0.0,
                "cls_loss": cls_loss_val,
                "cls_acc": cls_acc,
                "band_acc": band_acc,
                **per_ch,
            }

    @torch.no_grad()
    def embed(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor = None,
        gp_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode to latent vector (deterministic). Use for downstream tasks."""
        cls_out = self._encode_cls(x, pad_mask, images, gp_features)
        if self.mode == "ae":
            return self.bottleneck_proj(cls_out)
        elif self.mode == "vae":
            return self.mu_proj(cls_out)
        elif self.mode == "vqvae":
            z_e = self.pre_vq_proj(cls_out)
            z_q, _, _ = self.vq(z_e)
            return z_q

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor = None,
        gp_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Per-sample reconstruction error (anomaly score)."""
        z = self.embed(x, pad_mask, images, gp_features)
        cont_hat, band_logits = self.decode(z)
        L = x.shape[1]
        cont_hat, band_logits = cont_hat[:, :L], band_logits[:, :L]

        valid = ~pad_mask
        n_valid = valid.sum(dim=1).clamp(min=1).float()

        cont_err = ((cont_hat - x[..., :4]) ** 2 * valid.unsqueeze(-1)).sum(
            dim=(1, 2)
        ) / (n_valid * 4)
        band_target = x[..., 4:7].argmax(-1)
        B, L_act = band_target.shape
        ce = F.cross_entropy(
            band_logits.reshape(-1, 3), band_target.reshape(-1), reduction="none"
        ).reshape(B, L_act)
        band_err = (ce * valid).sum(dim=1) / n_valid

        return cont_err + band_err

    def compression_info(self) -> dict:
        """Return compression statistics for this model configuration."""
        raw_size = self.in_channels * self.max_len * 4  # float32
        if self.mode == "vqvae":
            import math

            bits_per_alert = math.ceil(math.log2(self.vq.num_codes))
            compressed_bytes = max(1, bits_per_alert // 8 + 1)
        else:
            compressed_bytes = self.latent_dim * 4
        return {
            "raw_bytes": raw_size,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": raw_size / compressed_bytes,
            "latent_dim": self.latent_dim,
            "in_channels": self.in_channels,
            "mode": self.mode,
        }
