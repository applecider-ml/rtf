"""
Perceiver IO Masked Autoencoder for Multi-Modal ZTF Alerts.

Tokenizes photometry, cutout images, and metadata into a shared sequence,
compresses through a learned latent bottleneck via cross-attention, and
trains via masked reconstruction of photometry tokens.

Architecture:
    Tokenizer:
        - Photometry: Linear(7, d_model) + Linear(30, d_model) + Time2Vec
        - Image: ImageTower → (1, d_model) token
        - Metadata: MLP → (1, d_model) token
        → concatenated to (B, L+2, d_model)

    Encoder:
        - Cross-attention: latent array (32, d_model) attends to input tokens
        - Self-attention: N layers on latent array
        → pooled to (B, d_model) → projected to (B, 128)

    Decoder (training only):
        - Cross-attention: position queries attend to latent array
        → reconstruct masked photometry tokens

    Loss: MSE(cont) + CE(band) on masked positions only.
"""

import torch
import torch.nn.functional as F
from torch import nn

from model import ImageTower, Time2Vec

# ---------------------------------------------------------------------------
# Core Attention Blocks
# ---------------------------------------------------------------------------


class CrossAttentionBlock(nn.Module):
    """Cross-attention: queries attend to key-value pairs from a different source.

    Used for:
        - Encoder: latent array (Q) attends to input tokens (KV)
        - Decoder: position queries (Q) attend to latent array (KV)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d_model)
            kv:    (B, N_kv, d_model)
            key_padding_mask: (B, N_kv) True = padded (ignored)
        Returns:
            (B, N_q, d_model)
        """
        # Pre-norm cross-attention
        q_normed = self.norm_q(query)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.attn(
            q_normed,
            kv_normed,
            kv_normed,
            key_padding_mask=key_padding_mask,
        )
        query = query + attn_out

        # Pre-norm feedforward
        query = query + self.ffn(self.norm_ff(query))
        return query


class SelfAttentionBlock(nn.Module):
    """Standard transformer self-attention block (pre-norm).

    Used for self-attention layers on the latent array.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → (B, N, d_model)"""
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm_ff(x))
        return x


# ---------------------------------------------------------------------------
# Perceiver Encoder
# ---------------------------------------------------------------------------


class PerceiverEncoder(nn.Module):
    """Cross-attention from latent array to inputs, then self-attention on latents.

    The latent array is a learned parameter (not derived from input), providing
    a fixed-size bottleneck regardless of input sequence length.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_latents: int = 32,
        n_cross: int = 2,
        n_self: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Learned latent array — the bottleneck
        self.latent_array = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        # Cross-attention layers (latents attend to input tokens)
        self.cross_layers = nn.ModuleList(
            [CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_cross)]
        )

        # Self-attention layers (latents attend to each other)
        self.self_layers = nn.ModuleList(
            [SelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_self)]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_tokens: (B, N_input, d_model)
            key_padding_mask: (B, N_input) True = padded
        Returns:
            latents: (B, n_latents, d_model)
        """
        B = input_tokens.shape[0]
        latents = self.latent_array.unsqueeze(0).expand(B, -1, -1)  # (B, 32, d_model)

        # Cross-attention: latents attend to input
        for cross_layer in self.cross_layers:
            latents = cross_layer(latents, input_tokens, key_padding_mask)

        # Self-attention: latents refine each other
        for self_layer in self.self_layers:
            latents = self_layer(latents)

        return self.norm(latents)


# ---------------------------------------------------------------------------
# Perceiver Decoder
# ---------------------------------------------------------------------------


class PerceiverDecoder(nn.Module):
    """Decode masked positions by cross-attending from position queries to latents.

    Only used during training for reconstruction loss. Discarded at inference.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        max_len: int = 300,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Learned position queries — one per possible sequence position
        self.position_queries = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # Cross-attention: position queries attend to encoder latents
        self.cross_attn = CrossAttentionBlock(d_model, n_heads, dropout)

        # Prediction heads
        self.head_cont = nn.Linear(d_model, 4)  # dt, dt_prev, flux, flux_err
        self.head_band = nn.Linear(d_model, 3)  # band logits (g, r, i)

    def forward(
        self,
        latents: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latents: (B, n_latents, d_model) from encoder
            mask_indices: (B, n_mask) indices of masked positions

        Returns:
            cont_pred: (B, n_mask, 4) continuous value predictions
            band_pred: (B, n_mask, 3) band classification logits
        """
        _B, _n_mask = mask_indices.shape

        # Gather position queries for masked positions
        # position_queries: (max_len, d_model) → index with mask_indices
        queries = self.position_queries[mask_indices]  # (B, n_mask, d_model)

        # Cross-attend: queries attend to latents
        decoded = self.cross_attn(queries, latents)  # (B, n_mask, d_model)

        cont_pred = self.head_cont(decoded)  # (B, n_mask, 4)
        band_pred = self.head_band(decoded)  # (B, n_mask, 3)

        return cont_pred, band_pred


# ---------------------------------------------------------------------------
# Perceiver Autoencoder (Full Model)
# ---------------------------------------------------------------------------


class PerceiverAutoencoder(nn.Module):
    """Multi-modal Perceiver IO masked autoencoder for ZTF alerts.

    Tokenizes photometry, images, and metadata into a shared sequence,
    masks 50% of photometry tokens, compresses through a latent bottleneck,
    and reconstructs masked tokens.

    Training:
        out = model(x, pad_mask, images)
        loss_dict = model.compute_loss(x, pad_mask, out)

    Inference:
        embedding = model.embed(x, pad_mask, images)  # (B, 128)
    """

    def __init__(
        self,
        # Input dimensions
        phot_channels: int = 7,
        meta_channels: int = 30,
        # Architecture
        d_model: int = 256,
        n_latents: int = 32,
        n_cross: int = 2,
        n_self: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        # Output
        embed_dim: int = 128,
        # Masking
        mask_ratio: float = 0.50,
        max_len: int = 300,
        # Image tower
        image_backbone: str = "simple",
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # ── Tokenizers ──

        # Photometry: project 7 channels to d_model
        self.phot_proj = nn.Linear(phot_channels, d_model)

        # Per-observation metadata: project 30 channels to d_model
        self.meta_obs_proj = nn.Linear(meta_channels, d_model)

        # Time encoding (reuse RTF's Time2Vec)
        self.time2vec = Time2Vec(d_model)

        # Static metadata summary token
        self.meta_summary_mlp = nn.Sequential(
            nn.Linear(meta_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Image token
        self.image_tower = ImageTower(d_model, dropout, backbone=image_backbone)

        # Learned modality embeddings
        self.modality_emb_phot = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_emb_img = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_emb_meta = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learned [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # ── Encoder ──
        self.encoder = PerceiverEncoder(
            d_model=d_model,
            n_latents=n_latents,
            n_cross=n_cross,
            n_self=n_self,
            n_heads=n_heads,
            dropout=dropout,
        )

        # ── Embedding projection ──
        self.embed_proj = nn.Linear(d_model, embed_dim)

        # ── Decoder (training only) ──
        self.decoder = PerceiverDecoder(
            d_model=d_model,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
        )

    def _tokenize(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw inputs into a sequence of tokens.

        Args:
            x: (B, L, 37) full alert sequence
            pad_mask: (B, L)
            images: (B, 3, 63, 63)

        Returns:
            tokens: (B, L+2, d_model)
            token_pad_mask: (B, L+2) — image/meta tokens are never padded
        """
        B, _L, _ = x.shape

        # Split channels
        x_phot = x[:, :, :7]  # (B, L, 7)
        x_meta = x[:, :, 7:]  # (B, L, 30)

        # Photometry tokens
        phot_tokens = self.phot_proj(x_phot)  # (B, L, d_model)
        phot_tokens = phot_tokens + self.meta_obs_proj(x_meta)  # add per-obs metadata
        phot_tokens = phot_tokens + self.time2vec(x[:, :, 0])  # add time encoding
        phot_tokens = phot_tokens + self.modality_emb_phot  # add modality ID

        # Image token
        img_emb = self.image_tower(images)  # (B, d_model)
        img_token = img_emb.unsqueeze(1)  # (B, 1, d_model)
        img_token = img_token + self.modality_emb_img

        # Metadata summary token
        static_meta = x_meta.mean(dim=1)  # (B, 30)
        meta_emb = self.meta_summary_mlp(static_meta)  # (B, d_model)
        meta_token = meta_emb.unsqueeze(1)  # (B, 1, d_model)
        meta_token = meta_token + self.modality_emb_meta

        # Concatenate: [phot_0, ..., phot_L-1, img, meta]
        tokens = torch.cat([phot_tokens, img_token, meta_token], dim=1)  # (B, L+2, d_model)

        # Padding mask: image and metadata tokens are never padded
        extra_mask = torch.zeros(B, 2, dtype=torch.bool, device=x.device)
        token_pad_mask = torch.cat([pad_mask, extra_mask], dim=1)  # (B, L+2)

        return tokens, token_pad_mask

    def _apply_masking(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor,
        L: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask a random subset of photometry tokens (positions 0..L-1).

        Args:
            tokens: (B, L+2, d_model)
            pad_mask: (B, L) original photometry padding mask
            L: sequence length (number of photometry tokens)

        Returns:
            masked_tokens: (B, L+2, d_model) with mask token replacing masked positions
            mask_indices: (B, max_n_mask) indices of masked photometry positions
            mask_valid: (B, max_n_mask) bool — True where mask_indices is a real
                        masked position (not filler from padding shorter samples)
        """
        B = tokens.shape[0]
        device = tokens.device

        # For each sample, count valid (non-padded) positions
        valid_counts = (~pad_mask).sum(dim=1)  # (B,)

        # Determine how many to mask per sample
        n_mask_per_sample = (valid_counts.float() * self.mask_ratio).long().clamp(min=1)
        max_n_mask = n_mask_per_sample.max().item()

        # Build mask indices per sample + validity tracker
        mask_indices = torch.zeros(B, max_n_mask, dtype=torch.long, device=device)
        mask_valid = torch.zeros(B, max_n_mask, dtype=torch.bool, device=device)
        for b in range(B):
            n_valid = valid_counts[b].item()
            n_mask = n_mask_per_sample[b].item()
            # Random permutation of valid indices, take first n_mask
            perm = torch.randperm(n_valid, device=device)[:n_mask]
            mask_indices[b, :n_mask] = perm
            mask_valid[b, :n_mask] = True
            # Remaining positions stay 0 (index 0) but mask_valid is False
            # so they won't contribute to loss

        # Replace masked positions with learned [MASK] token
        masked_tokens = tokens.clone()
        # Only replace positions that are actually masked (not filler)
        for b in range(B):
            n_mask = n_mask_per_sample[b].item()
            masked_tokens[b, mask_indices[b, :n_mask]] = self.mask_token

        return masked_tokens, mask_indices, mask_valid

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> dict:
        """Full forward pass with masking and reconstruction.

        Args:
            x: (B, L, 37)
            pad_mask: (B, L)
            images: (B, 3, 63, 63)

        Returns:
            dict with 'latents', 'mask_indices', 'mask_valid', 'cont_pred', 'band_pred'
        """
        _B, L, _ = x.shape

        # Tokenize
        tokens, token_pad_mask = self._tokenize(x, pad_mask, images)

        # Mask photometry tokens
        masked_tokens, mask_indices, mask_valid = self._apply_masking(tokens, pad_mask, L)

        # Encode
        latents = self.encoder(masked_tokens, token_pad_mask)  # (B, n_latents, d_model)

        # Decode masked positions
        cont_pred, band_pred = self.decoder(latents, mask_indices)

        return {
            "latents": latents,
            "mask_indices": mask_indices,
            "mask_valid": mask_valid,
            "cont_pred": cont_pred,
            "band_pred": band_pred,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        forward_out: dict,
    ) -> dict:
        """Compute reconstruction loss on masked positions only.

        Args:
            x: (B, L, 37) original (unmasked) input
            pad_mask: (B, L) padding mask
            forward_out: output from forward()

        Returns:
            dict with 'total_loss', 'recon_cont', 'recon_band', 'band_acc'
        """
        mask_indices = forward_out["mask_indices"]  # (B, n_mask)
        mask_valid = forward_out["mask_valid"]  # (B, n_mask) True = real mask
        cont_pred = forward_out["cont_pred"]  # (B, n_mask, 4)
        band_pred = forward_out["band_pred"]  # (B, n_mask, 3)
        B, n_mask = mask_indices.shape

        # Gather targets at masked positions
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, n_mask)

        cont_target = x[batch_idx, mask_indices, :4]  # (B, n_mask, 4)
        band_target = x[batch_idx, mask_indices, 4:7].argmax(-1)  # (B, n_mask)

        # Use mask_valid: only count loss on positions that were actually masked
        # (not filler indices from padding shorter samples to max_n_mask)
        n_valid = mask_valid.sum().clamp(min=1)

        # Continuous reconstruction (MSE on valid masked positions)
        cont_err = ((cont_pred - cont_target) ** 2) * mask_valid.unsqueeze(-1)
        recon_cont = cont_err.sum() / (n_valid * 4)

        # Band reconstruction (CE on valid masked positions)
        band_ce = F.cross_entropy(
            band_pred.reshape(-1, 3), band_target.reshape(-1), reduction="none"
        )
        band_ce = (band_ce.reshape(B, n_mask) * mask_valid).sum() / n_valid
        recon_band = band_ce

        # Band accuracy
        band_acc = (
            (band_pred.argmax(-1) == band_target) * mask_valid
        ).sum().item() / n_valid.item()

        total_loss = recon_cont + recon_band

        return {
            "total_loss": total_loss,
            "recon_cont": recon_cont.item(),
            "recon_band": recon_band.item(),
            "band_acc": band_acc,
        }

    @torch.no_grad()
    def embed(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Produce 128D embedding (no masking, no decoder).

        Compatible with RTF's embed() interface.

        Returns:
            (B, 128) embedding
        """
        # Tokenize (no masking at inference)
        tokens, token_pad_mask = self._tokenize(x, pad_mask, images)

        # Encode
        latents = self.encoder(tokens, token_pad_mask)  # (B, n_latents, d_model)

        # Pool latents → single vector
        pooled = latents.mean(dim=1)  # (B, d_model)

        # Project to embedding space
        return self.embed_proj(pooled)  # (B, 128)

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample reconstruction error (for anomaly scoring).

        Masks ALL photometry tokens and measures how well they're reconstructed.
        Higher error = more unusual/anomalous.

        Returns:
            (B,) reconstruction error per sample
        """
        B, _L, _ = x.shape

        # Tokenize
        tokens, token_pad_mask = self._tokenize(x, pad_mask, images)

        # Mask ALL photometry tokens (aggressive — for anomaly scoring)
        all_masked = tokens.clone()
        valid = ~pad_mask  # (B, L)
        for b in range(B):
            n_valid = valid[b].sum().item()
            all_masked[b, :n_valid] = self.mask_token

        # Build mask_indices: all valid positions
        max_valid = valid.sum(dim=1).max().item()
        mask_indices = torch.zeros(B, max_valid, dtype=torch.long, device=x.device)
        for b in range(B):
            n_v = valid[b].sum().item()
            mask_indices[b, :n_v] = torch.arange(n_v, device=x.device)
            if n_v < max_valid:
                mask_indices[b, n_v:] = n_v - 1  # pad with last valid

        # Encode + decode
        latents = self.encoder(all_masked, token_pad_mask)
        cont_pred, _band_pred = self.decoder(latents, mask_indices)

        # Per-sample MSE on continuous channels
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, max_valid)
        cont_target = x[batch_idx, mask_indices, :4]
        valid_mask = valid[batch_idx, mask_indices]

        per_sample_err = ((cont_pred - cont_target) ** 2).mean(dim=-1)  # (B, max_valid)
        per_sample_err = (per_sample_err * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(
            min=1
        )

        return per_sample_err  # (B,)
