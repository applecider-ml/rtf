"""
Export RTF model to ONNX format for BOOM deployment.

Produces two ONNX graphs:
  - rtf_embed.onnx:  x, pad_mask, images  →  embedding (1, 128)
  - rtf_recon.onnx:  x, pad_mask, images  →  recon_error (1,)

Usage:
    python export_onnx.py \
        --model-path /path/to/best_model.pt \
        --output-dir /path/to/output

Verification:
    The script automatically runs both ONNX graphs on a random input
    and compares against PyTorch outputs to ensure correctness.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model import LightCurveCompressor

MAX_LEN = 257
IN_CHANNELS = 37


# ── Wrapper modules for clean ONNX graphs ──────────────────────────────

class EmbedWrapper(nn.Module):
    """Wraps model.embed() into a single-output module for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, pad_mask, images):
        cls_out = self.model._encode_cls(x, pad_mask, images, gp_features=None)
        # AE mode: bottleneck_proj
        return self.model.bottleneck_proj(cls_out)


class ReconWrapper(nn.Module):
    """Wraps model.reconstruction_error() for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, pad_mask, images):
        # Encode
        cls_out = self.model._encode_cls(x, pad_mask, images, gp_features=None)
        z = self.model.bottleneck_proj(cls_out)

        # Decode
        cont_hat, band_logits = self.model.decode(z)
        L = x.shape[1]
        cont_hat = cont_hat[:, :L]
        band_logits = band_logits[:, :L]

        # Reconstruction error
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


def load_model(model_path, device="cpu"):
    """Load the trained RTF model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_kwargs = checkpoint.get("model_kwargs", {})
    else:
        state_dict = checkpoint
        model_kwargs = {}

    if not model_kwargs:
        model_kwargs = {
            "mode": "ae",
            "latent_dim": 128,
            "in_channels": IN_CHANNELS,
            "d_model": 256,
        }

    # Ensure no GP features in ONNX export (they are optional and rarely used)
    model_kwargs.pop("gp_dim", None)
    model_kwargs["gp_dim"] = 0

    model = LightCurveCompressor(**model_kwargs)

    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v
    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model, model_kwargs


def export_graph(wrapper, name, output_dir, dummy_x, dummy_mask, dummy_img):
    """Export a single ONNX graph."""
    path = os.path.join(output_dir, f"{name}.onnx")
    print(f"\n  Exporting {name}...")

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_mask, dummy_img),
        path,
        opset_version=17,
        input_names=["x", "pad_mask", "images"],
        output_names=["output"],
        dynamic_axes={
            "x": {0: "batch"},
            "pad_mask": {0: "batch"},
            "images": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    fsize = os.path.getsize(path) / 1e6
    print(f"  Saved: {path} ({fsize:.1f} MB)")
    return path


def verify_onnx(onnx_path, wrapper, dummy_x, dummy_mask, dummy_img, name):
    """Verify ONNX output matches PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed, skipping verification")
        return

    # PyTorch reference
    with torch.no_grad():
        pt_out = wrapper(dummy_x, dummy_mask, dummy_img).numpy()

    # ONNX inference — auto-detect input names (ONNX may rename bool inputs)
    sess = ort.InferenceSession(onnx_path)
    onnx_inputs = sess.get_inputs()
    print(f"  ONNX input names: {[inp.name for inp in onnx_inputs]}")

    # Map positional args to whatever ONNX named them
    numpy_inputs = [dummy_x.numpy(), dummy_mask.numpy(), dummy_img.numpy()]
    feed = {}
    for inp, val in zip(onnx_inputs, numpy_inputs):
        feed[inp.name] = val

    ort_out = sess.run(None, feed)[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"  {name} max diff: {diff:.2e} (threshold: 1e-4)")
    if diff < 1e-4:
        print(f"  ✓ {name} PASSED verification")
    else:
        print(f"  ✗ {name} FAILED — outputs differ by {diff:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Export RTF model to ONNX")
    parser.add_argument(
        "--model-path", required=True, help="Path to best_model.pt"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for ONNX files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== RTF ONNX Export ===")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {args.output_dir}")

    # Load model
    print("\n[1/4] Loading model...")
    model, kwargs = load_model(args.model_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params")
    print(f"  mode={kwargs.get('mode')}, latent_dim={kwargs.get('latent_dim')}")
    print(f"  in_channels={kwargs.get('in_channels')}, d_model={kwargs.get('d_model')}")

    # Create dummy inputs
    print("\n[2/4] Creating dummy inputs...")
    dummy_x = torch.randn(1, MAX_LEN, IN_CHANNELS)
    dummy_mask = torch.zeros(1, MAX_LEN, dtype=torch.bool)
    # Simulate padding: last 50 positions are padding
    dummy_mask[0, 207:] = True
    dummy_img = torch.randn(1, 3, 63, 63)
    print(f"  x: {dummy_x.shape}, pad_mask: {dummy_mask.shape}, images: {dummy_img.shape}")

    # Export embed graph
    print("\n[3/4] Exporting ONNX graphs...")
    embed_wrapper = EmbedWrapper(model)
    embed_wrapper.eval()
    embed_path = export_graph(
        embed_wrapper, "rtf_embed", args.output_dir, dummy_x, dummy_mask, dummy_img
    )

    # Export recon graph
    recon_wrapper = ReconWrapper(model)
    recon_wrapper.eval()
    recon_path = export_graph(
        recon_wrapper, "rtf_recon", args.output_dir, dummy_x, dummy_mask, dummy_img
    )

    # Verify
    print("\n[4/4] Verifying ONNX outputs...")
    verify_onnx(embed_path, embed_wrapper, dummy_x, dummy_mask, dummy_img, "embed")
    verify_onnx(recon_path, recon_wrapper, dummy_x, dummy_mask, dummy_img, "recon")

    print("\n=== DONE ===")
    print(f"  rtf_embed.onnx → {embed_path}")
    print(f"  rtf_recon.onnx → {recon_path}")
    print("\n  Copy these files to boom/data/models/ for deployment.")


if __name__ == "__main__":
    main()
