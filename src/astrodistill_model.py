"""
AstroDistill: Data2Vec-Style Self-Distillation for ZTF Transient Alerts.

A teacher network sees the complete (unmasked) multi-modal input.
A student network sees a masked version. The student learns to predict
the teacher's contextualized latent representations at masked positions.

This is the first application of Data2Vec-style self-distillation to
astronomical transient alert data.

Key insight: Unlike contrastive learning (TransientBind), self-distillation
CANNOT collapse because:
  - L2 loss has no batch-level interaction (no negatives)
  - Teacher targets are unique per sample (unique input + instance norm)
  - The student is forced to produce unique predictions

Architecture:
    Tokenizer (same as Perceiver IO):
        - Photometry: Linear(7, d_model) + Linear(30, d_model) + Time2Vec
        - Image: ImageTower → (1, d_model) token
        - Metadata: MLP → (1, d_model) token
        → concatenated to (B, L+2, d_model)

    Student encoder: 4-layer Transformer (standard, NOT Perceiver bottleneck)
    Teacher encoder: EMA copy of student (no gradient)
    Prediction MLP: 2-layer (d_model → hidden → d_model)

    Loss: L2(student_prediction, instance_norm(teacher_target)) at masked positions
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from model import ImageTower, Time2Vec

# ---------------------------------------------------------------------------
# Transformer Encoder (returns all layer outputs for top-K averaging)
# ---------------------------------------------------------------------------


class TransformerLayerStack(nn.Module):
    """Standard Transformer encoder that can return intermediate layer outputs.

    Unlike nn.TransformerEncoder, this stores outputs from each layer
    so we can average the top-K layers for teacher targets (Data2Vec).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Standard forward — returns final layer output only.

        Args:
            x: (B, N, d_model)
            src_key_padding_mask: (B, N) True = padded

        Returns:
            (B, N, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)

    def forward_all_layers(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> list[torch.Tensor]:
        """Returns output from EVERY layer (for top-K averaging).

        Args:
            x: (B, N, d_model)
            src_key_padding_mask: (B, N) True = padded

        Returns:
            list of n_layers tensors, each (B, N, d_model)
        """
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            layer_outputs.append(x)
        # Apply final norm to last layer only (for consistency with forward())
        layer_outputs[-1] = self.norm(layer_outputs[-1])
        return layer_outputs


# ---------------------------------------------------------------------------
# Prediction MLP (used only during training, discarded at inference)
# ---------------------------------------------------------------------------


class PredictionMLP(nn.Module):
    """2-layer MLP that predicts teacher targets from student hidden states.

    Input:  (B, n_mask, d_model)
    Output: (B, n_mask, d_model)
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# AstroDistill: Full Model
# ---------------------------------------------------------------------------


class AstroDistill(nn.Module):
    """Data2Vec-style self-distillation for multi-modal ZTF alerts.

    Training:
        out = model(x, pad_mask, images)
        loss_dict = out  # forward() returns the loss directly
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()
        model.update_teacher()  # EMA update after each step

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
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        # Output
        embed_dim: int = 128,
        # Masking
        mask_ratio: float = 0.50,
        max_len: int = 300,
        # Image tower
        image_backbone: str = "simple",
        # Data2Vec specific
        top_k_layers: int = 2,
        pred_hidden: int = 512,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.top_k_layers = top_k_layers
        self.ema_decay = ema_decay

        # ── Tokenizers (IDENTICAL to Perceiver IO) ──

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

        # ── Student encoder ──
        self.student_encoder = TransformerLayerStack(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # ── Teacher encoder (deep copy of student, NO gradient) ──
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        # ── Prediction MLP (student → teacher target space) ──
        self.prediction_mlp = PredictionMLP(d_model, pred_hidden)

        # ── Embedding projection (inference only) ──
        self.embed_proj = nn.Linear(d_model, embed_dim)

    # ------------------------------------------------------------------
    # Tokenization (COPIED VERBATIM from perceiver_model.py)
    # ------------------------------------------------------------------

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
        x_phot = x[:, :, :7]     # (B, L, 7)
        x_meta = x[:, :, 7:]     # (B, L, 30)

        # Photometry tokens
        phot_tokens = self.phot_proj(x_phot)                    # (B, L, d_model)
        phot_tokens = phot_tokens + self.meta_obs_proj(x_meta)  # add per-obs metadata
        phot_tokens = phot_tokens + self.time2vec(x[:, :, 0])   # add time encoding
        phot_tokens = phot_tokens + self.modality_emb_phot      # add modality ID

        # Image token
        img_emb = self.image_tower(images)                # (B, d_model)
        img_token = img_emb.unsqueeze(1)                  # (B, 1, d_model)
        img_token = img_token + self.modality_emb_img

        # Metadata summary token
        static_meta = x_meta.mean(dim=1)                  # (B, 30)
        meta_emb = self.meta_summary_mlp(static_meta)     # (B, d_model)
        meta_token = meta_emb.unsqueeze(1)                # (B, 1, d_model)
        meta_token = meta_token + self.modality_emb_meta

        # Concatenate: [phot_0, ..., phot_L-1, img, meta]
        tokens = torch.cat([phot_tokens, img_token, meta_token], dim=1)  # (B, L+2, d_model)

        # Padding mask: image and metadata tokens are never padded
        extra_mask = torch.zeros(B, 2, dtype=torch.bool, device=x.device)
        token_pad_mask = torch.cat([pad_mask, extra_mask], dim=1)  # (B, L+2)

        return tokens, token_pad_mask

    # ------------------------------------------------------------------
    # Masking (COPIED VERBATIM from perceiver_model.py)
    # ------------------------------------------------------------------

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

        # Replace masked positions with learned [MASK] token
        masked_tokens = tokens.clone()
        for b in range(B):
            n_mask = n_mask_per_sample[b].item()
            masked_tokens[b, mask_indices[b, :n_mask]] = self.mask_token

        return masked_tokens, mask_indices, mask_valid

    # ------------------------------------------------------------------
    # Instance normalization for teacher targets (prevents collapse)
    # ------------------------------------------------------------------

    @staticmethod
    def _instance_normalize(x: torch.Tensor) -> torch.Tensor:
        """Normalize each sample independently to zero-mean, unit-variance.

        This is CRITICAL — without it, the teacher can trivially produce
        constant targets across all positions, and the student learns nothing.

        Args:
            x: (B, N, D)
        Returns:
            (B, N, D) normalized per-sample
        """
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True)
        return (x - mean) / (var.sqrt() + 1e-6)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> dict:
        """Full forward pass: teacher + student + loss computation.

        Args:
            x: (B, L, 37)
            pad_mask: (B, L)
            images: (B, 3, 63, 63)

        Returns:
            dict with 'total_loss' (Tensor, requires grad), 'loss_value' (float)
        """
        B, L, _ = x.shape

        # ── Step 1: Tokenize (shared between teacher and student) ──
        tokens, token_pad_mask = self._tokenize(x, pad_mask, images)

        # ── Step 2: Teacher forward (full input, no masking, no gradient) ──
        with torch.no_grad():
            teacher_all_layers = self.teacher_encoder.forward_all_layers(
                tokens, src_key_padding_mask=token_pad_mask,
            )
            # Average top-K layers for smooth targets
            k = self.top_k_layers
            teacher_targets = torch.stack(teacher_all_layers[-k:]).mean(dim=0)
            # (B, L+2, d_model)

            # Instance normalize targets (per-sample zero-mean, unit-var)
            teacher_targets = self._instance_normalize(teacher_targets)

        # ── Step 3: Apply masking to create student input ──
        masked_tokens, mask_indices, mask_valid = self._apply_masking(
            tokens, pad_mask, L,
        )

        # ── Step 4: Student forward (masked input, gradient flows) ──
        student_out = self.student_encoder(
            masked_tokens, src_key_padding_mask=token_pad_mask,
        )
        # student_out: (B, L+2, d_model)

        # ── Step 5: Extract student hidden states at masked positions ──
        n_mask = mask_indices.shape[1]
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, n_mask)
        student_at_masked = student_out[batch_idx, mask_indices]  # (B, n_mask, d_model)

        # ── Step 6: Predict teacher targets ──
        predictions = self.prediction_mlp(student_at_masked)  # (B, n_mask, d_model)

        # ── Step 7: Extract teacher targets at masked positions ──
        targets = teacher_targets[batch_idx, mask_indices]  # (B, n_mask, d_model)

        # ── Step 8: L2 loss (only on valid masked positions) ──
        per_position_loss = ((predictions - targets) ** 2).mean(dim=-1)  # (B, n_mask)
        n_valid = mask_valid.sum().clamp(min=1)
        total_loss = (per_position_loss * mask_valid).sum() / n_valid

        return {
            "total_loss": total_loss,
            "loss_value": total_loss.item(),
        }

    # ------------------------------------------------------------------
    # Teacher EMA update (call AFTER each optimizer step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, decay: float | None = None):
        """Update teacher weights as EMA of student weights.

        Args:
            decay: EMA decay factor. If None, uses self.ema_decay.
                   Pass a cosine-scheduled value for better training dynamics.
        """
        if decay is None:
            decay = self.ema_decay
        for t_param, s_param in zip(
            self.teacher_encoder.parameters(),
            self.student_encoder.parameters(),
        ):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

    # ------------------------------------------------------------------
    # Inference: embed() — matches RTF interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Produce 128D embedding (no masking, no teacher, no prediction MLP).

        Uses the student encoder only. Mean-pools over non-padded positions.
        Compatible with RTF's embed() interface.

        Returns:
            (B, 128) embedding
        """
        tokens, token_pad_mask = self._tokenize(x, pad_mask, images)

        # No masking at inference
        h = self.student_encoder(tokens, src_key_padding_mask=token_pad_mask)
        # h: (B, L+2, d_model)

        # Mean-pool over non-padded positions
        valid = ~token_pad_mask  # (B, L+2) True = valid
        valid_f = valid.unsqueeze(-1).float()  # (B, L+2, 1)
        pooled = (h * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1)
        # pooled: (B, d_model)

        return self.embed_proj(pooled)  # (B, 128)

    # ------------------------------------------------------------------
    # Inference: reconstruction_error() — matches RTF interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample anomaly score based on student-teacher disagreement.

        Masks ALL photometry tokens and measures how well the student
        predicts the teacher's representations. Higher = more anomalous.

        Returns:
            (B,) reconstruction error per sample
        """
        B, L, _ = x.shape

        # Tokenize
        tokens, token_pad_mask = self._tokenize(x, pad_mask, images)

        # Teacher: full input
        teacher_all_layers = self.teacher_encoder.forward_all_layers(
            tokens, src_key_padding_mask=token_pad_mask,
        )
        k = self.top_k_layers
        teacher_targets = torch.stack(teacher_all_layers[-k:]).mean(dim=0)
        teacher_targets = self._instance_normalize(teacher_targets)

        # Student: mask ALL photometry tokens
        all_masked = tokens.clone()
        valid = ~pad_mask  # (B, L)
        for b in range(B):
            n_valid = valid[b].sum().item()
            all_masked[b, :n_valid] = self.mask_token

        student_out = self.student_encoder(
            all_masked, src_key_padding_mask=token_pad_mask,
        )

        # Measure L2 distance at all valid photometry positions
        # (not image/meta tokens — those weren't masked)
        per_position_err = ((student_out[:, :L] - teacher_targets[:, :L]) ** 2).mean(dim=-1)
        # per_position_err: (B, L)
        per_sample_err = (per_position_err * valid.float()).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        return per_sample_err  # (B,)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compression_info(self) -> dict:
        """Compression stats for logging (matches RTF interface)."""
        return {
            "latent_dim": self.embed_dim,
            "compressed_bytes": self.embed_dim * 4,
            "compression_ratio": 0.0,
        }
