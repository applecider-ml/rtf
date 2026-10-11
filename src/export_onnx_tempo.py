#!/usr/bin/env python3
"""Export TEMPO EvidentialClassifier to ONNX.

Usage (on Delta):
    python export_onnx_tempo.py \
        --checkpoint /path/to/best_evidential.pt \
        --output-dir /path/to/onnx/

The script adds the TEMPO codebase to sys.path so it can import
photometry_edl. Set TEMPO_ROOT env var if needed.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Add TEMPO codebase to path so we can import photometry_edl
# ---------------------------------------------------------------------------
TEMPO_ROOT = os.environ.get(
    "TEMPO_ROOT",
    "/projects/bcrv/ffontinelenunes/transformer/tempo",
)
if TEMPO_ROOT not in sys.path:
    sys.path.insert(0, TEMPO_ROOT)

from photometry_edl.models.transformer import (
    EventTransformerEncoder,
    EvidentialClassifier,
)

# ---------------------------------------------------------------------------
# Model config from the best checkpoint
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "d_model": 192,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.0,  # inference mode, no dropout
    "band_mode": "embed",
    "band_embed_dim": 8,
    "time_encoding": "dt_first",
    "ff_mult": 4,
}

CLASSIFIER_CONFIG = {
    "num_classes": 5,
    "dropout": 0.0,  # inference mode
    "pool": "cls",
    "use_global_features": True,
    "global_dim": 24,
    "global_hidden_dim": 192,
}

MAX_LEN = 257
CLASS_NAMES = ["SNI", "SNII", "TDE", "AGN", "CV"]


# ---------------------------------------------------------------------------
# Export wrapper — bypasses runtime validation for clean ONNX tracing
# ---------------------------------------------------------------------------
class TempoExportWrapper(nn.Module):
    """Wraps EvidentialClassifier for ONNX export.

    Bypasses runtime shape/value assertions that break torch.onnx.export.
    """

    def __init__(self, model: EvidentialClassifier):
        super().__init__()
        self.encoder = model.encoder
        self.use_global = model.use_global_features
        self.global_mlp = model.global_mlp
        self.head = model.head

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass without validation checks.

        Args:
            x: (B, L, 5) normalized event tensor
            pad_mask: (B, L) boolean, True = padding
            global_features: (B, 24) physics features

        Returns:
            evidence: (B, 5) non-negative evidence vector
        """
        # --- Encoder (inline to avoid validation) ---
        enc = self.encoder
        B = x.shape[0]

        # Token projection
        h = enc.cont_proj(x[..., :4]) + enc._band_term(x)
        h = h + enc.dropout(enc._time_term(x))

        # Prepend CLS token
        tok = enc.cls_tok.expand(B, 1, -1)
        h = torch.cat([tok, h], dim=1)
        pad_full = torch.cat(
            [torch.zeros(B, 1, device=pad_mask.device, dtype=torch.bool), pad_mask],
            dim=1,
        )

        # Transformer
        z = enc.encoder(h, src_key_padding_mask=pad_full)
        z = enc.norm(z)

        # --- CLS pooling ---
        pooled = z[:, 0]

        # --- Global features ---
        if self.use_global:
            g = self.global_mlp(global_features.float())
            pooled = torch.cat([pooled, g], dim=-1)

        # --- Classification head ---
        logits = self.head(pooled)
        evidence = F.softplus(logits)
        return evidence


# ---------------------------------------------------------------------------
# Build model from config + checkpoint
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str) -> EvidentialClassifier:
    """Reconstruct model and load state dict."""
    # Build encoder with training dropout (needed to load state dict)
    encoder = EventTransformerEncoder(
        d_model=192,
        n_heads=8,
        n_layers=6,
        dropout=0.3,  # match training config for state dict shape
        band_mode="embed",
        band_embed_dim=8,
        time_encoding="dt_first",
        ff_mult=4,
    )
    model = EvidentialClassifier(
        encoder,
        num_classes=5,
        dropout=0.3,  # match training config
        pool="cls",
        use_global_features=True,
        global_dim=24,
        global_hidden_dim=192,
    )

    # Load state dict (EMA weights already applied before saving)
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params")
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(model, output_dir: Path):
    """Export TEMPO classifier to ONNX."""
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = TempoExportWrapper(model)
    wrapper.eval()

    # Dummy inputs
    dummy_x = torch.randn(1, MAX_LEN, 5)
    dummy_mask = torch.zeros(1, MAX_LEN, dtype=torch.bool)
    dummy_global = torch.randn(1, 24)

    print(f"  x: {dummy_x.shape}, pad_mask: {dummy_mask.shape}, global: {dummy_global.shape}")

    onnx_path = output_dir / "tempo_classifier.onnx"

    print("\n  Exporting tempo_classifier...")
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_mask, dummy_global),
        str(onnx_path),
        opset_version=17,
        input_names=["x", "pad_mask", "global_features"],
        output_names=["evidence"],
        dynamic_axes=None,  # fixed shape
    )

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path, wrapper, dummy_x, dummy_mask, dummy_global


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_onnx(onnx_path, wrapper, dummy_x, dummy_mask, dummy_global):
    """Verify ONNX output matches PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed, skipping verification")
        return

    # PyTorch reference
    with torch.no_grad():
        pt_out = wrapper(dummy_x, dummy_mask, dummy_global).numpy()

    # ONNX inference — auto-detect input names
    sess = ort.InferenceSession(str(onnx_path))
    onnx_inputs = sess.get_inputs()
    print(f"  ONNX input names: {[inp.name for inp in onnx_inputs]}")

    numpy_inputs = [dummy_x.numpy(), dummy_mask.numpy(), dummy_global.numpy()]
    feed = {}
    for inp, val in zip(onnx_inputs, numpy_inputs):
        feed[inp.name] = val

    ort_out = sess.run(None, feed)[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"  max diff: {diff:.2e} (threshold: 1e-4)")
    if diff < 1e-4:
        print("  ✓ PASSED verification")
    else:
        print(f"  ✗ FAILED — outputs differ by {diff:.6f}")

    # Print sample output
    print(f"\n  Sample evidence: {pt_out[0]}")
    alpha = pt_out[0] + 1.0
    probs = alpha / alpha.sum()
    for name, p in zip(CLASS_NAMES, probs):
        print(f"    {name}: {p:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export TEMPO to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="/work/nvme/bcrv/ffontinelenunes/tempo_runs/"
        "sota_sota_push_earlyuq_2026_04_15_185642/d_train/"
        "p26_tv5_margin_refine_012_006/best_evidential.pt",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ckpt.parent / "onnx"

    print("\n=== TEMPO ONNX Export ===")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Output: {out_dir}")
    print(f"  Classes: {CLASS_NAMES}")

    print("\n[1/3] Loading model...")
    model = load_model(str(ckpt))

    print("\n[2/3] Exporting ONNX graph...")
    onnx_path, wrapper, dx, dm, dg = export_onnx(model, out_dir)

    print("\n[3/3] Verifying ONNX output...")
    verify_onnx(onnx_path, wrapper, dx, dm, dg)

    print("\n=== DONE ===")
    print(f"  tempo_classifier.onnx → {onnx_path}")
    print("\n  Copy to boom/data/models/ for deployment.")


if __name__ == "__main__":
    main()
