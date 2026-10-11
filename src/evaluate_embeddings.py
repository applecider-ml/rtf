"""
Extract full 128-dim embeddings + per-component loss breakdown for Fritz sources.

For each Fritz source, saves:
  - 128-dim embedding vector
  - recon_cont (continuous channel MSE)
  - recon_band (band CE)
  - band_accuracy
  - overall recon_error

Usage:
    python evaluate_embeddings.py \
        --model-path /path/to/best_model.pt \
        --fritz-dir /path/to/fritz_sources \
        --output fritz_embeddings.npz
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import LightCurveCompressor


def load_model(model_path, device):
    """Load the frozen RTF autoencoder."""
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
            "in_channels": 37,
            "d_model": 256,
        }
    model = LightCurveCompressor(**model_kwargs)

    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v
    model.load_state_dict(clean_state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def infer_decomposed(model, x_np, image_np, device):
    """Run inference and decompose loss into recon_cont + recon_band.

    Returns:
        embedding: (128,) numpy array
        recon_cont: float (MSE on 4 continuous channels)
        recon_band: float (CE on band prediction)
        band_acc: float (band prediction accuracy)
        recon_error: float (overall reconstruction error from model)
    """
    MAX_LEN = 257
    if x_np.shape[0] > MAX_LEN:
        x_np = x_np[:MAX_LEN]

    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    pad_mask = torch.zeros(1, x.shape[1], dtype=torch.bool).to(device)
    images = torch.from_numpy(image_np).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get embedding
        emb = model.embed(x, pad_mask, images)  # (1, 128)
        # Get overall recon error
        recon_err = model.reconstruction_error(x, pad_mask, images)  # (1,)

        # Get forward output for loss decomposition
        out = model(x, pad_mask, images)
        # AE mode: out = (cont_hat, band_logits, z)
        cont_hat, band_logits, _z = out

        # Compute recon_cont (MSE on continuous channels)
        valid = ~pad_mask  # (1, L)
        n_valid = valid.sum().clamp(min=1)
        cont_target = x[..., :4]  # dt, dt_prev, logflux, logflux_err
        mse = ((cont_hat - cont_target) ** 2) * valid.unsqueeze(-1)
        recon_cont_val = (mse.sum() / (n_valid * 4)).item()

        # Compute recon_band (CE on band prediction)
        band_target = x[..., 4:7].argmax(-1)  # (1, L)
        B, L = pad_mask.shape
        ce = F.cross_entropy(band_logits.reshape(-1, 3), band_target.reshape(-1), reduction="none")
        recon_band_val = ((ce.reshape(B, L) * valid).sum() / n_valid).item()

        # Band accuracy
        band_acc = ((band_logits.argmax(-1) == band_target) * valid).sum().item() / n_valid.item()

    return (
        emb[0].cpu().numpy(),
        recon_cont_val,
        recon_band_val,
        band_acc,
        recon_err[0].item(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract Fritz embeddings + per-component loss decomposition"
    )
    parser.add_argument("--model-path", required=True, help="Path to best_model.pt")
    parser.add_argument("--fritz-dir", required=True, help="Fritz sources NPZ dir")
    parser.add_argument("--output", default="fritz_embeddings.npz", help="Output path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Fritz Embedding Extraction + Loss Decomposition ===")
    print(f"Device: {device}")

    # Load model
    print("\n[1/3] Loading frozen model...")
    model = load_model(args.model_path, device)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")

    # Find Fritz NPZ files
    print(f"\n[2/3] Scanning {args.fritz_dir}...")
    npz_files = sorted(Path(args.fritz_dir).glob("*.npz"))
    npz_files = [f for f in npz_files if f.stem != "fritz_manifest"]
    print(f"  Found {len(npz_files)} sources")

    # Extract
    print("\n[3/3] Extracting embeddings + loss components...")
    obj_ids = []
    embeddings = []
    recon_conts = []
    recon_bands = []
    band_accs = []
    recon_errors = []
    n_detections = []

    t0 = time.time()
    for idx, npz_path in enumerate(npz_files):
        obj_id = npz_path.stem
        try:
            data = np.load(npz_path, allow_pickle=True)
            x = data["x"]  # (L, 37)
            images = data["images"]  # (3, 63, 63)

            if x.shape[0] < 3:
                continue

            emb, rc, rb, ba, re = infer_decomposed(model, x, images, device)

            obj_ids.append(obj_id)
            embeddings.append(emb)
            recon_conts.append(rc)
            recon_bands.append(rb)
            band_accs.append(ba)
            recon_errors.append(re)
            n_detections.append(x.shape[0])

        except Exception as e:
            print(f"  [{idx + 1}] {obj_id}: ERROR - {e}")
            continue

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx + 1}/{len(npz_files)}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\n  Done: {len(obj_ids)} sources in {elapsed:.1f}s")

    # Save
    embeddings_arr = np.array(embeddings)
    print(f"  Embeddings shape: {embeddings_arr.shape}")
    print(f"  recon_cont: mean={np.mean(recon_conts):.6f}, std={np.std(recon_conts):.6f}")
    print(f"  recon_band: mean={np.mean(recon_bands):.6f}, std={np.std(recon_bands):.6f}")
    print(f"  band_acc:   mean={np.mean(band_accs):.3f}")
    print(f"  recon_err:  mean={np.mean(recon_errors):.6f}")

    np.savez_compressed(
        args.output,
        obj_ids=np.array(obj_ids),
        embeddings=embeddings_arr,
        recon_cont=np.array(recon_conts),
        recon_band=np.array(recon_bands),
        band_acc=np.array(band_accs),
        recon_error=np.array(recon_errors),
        n_detections=np.array(n_detections),
    )
    fsize = os.path.getsize(args.output)
    print(f"\n  Saved: {args.output} ({fsize / 1e6:.1f} MB)")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
