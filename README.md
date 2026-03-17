# alert-compression

Latent-space compression of ZTF photometry alerts using transformer autoencoders. Part of the [BOOM](https://github.com/boom-astro) project and the AppleCiDEr pipeline.

## Motivation

LSST will produce ~10M alerts/night. Current brokers filter this to a small subset, discarding most of the stream. This package explores compressing the full alert stream into fixed-length latent vectors that preserve astrophysically meaningful information, enabling:

- **Alert distribution at ~1 MBps** (streaming HD video bandwidth) instead of the full ~100 MBps raw stream
- **Similarity search** via cosine distance in latent space
- **Anomaly detection** via reconstruction error
- **Downstream classification** from latent vectors without access to raw data

## Approach

We train transformer-based autoencoders on ~18K labeled ZTF transient light curves and compare three bottleneck architectures:

| Architecture | Bottleneck | Regularization |
|---|---|---|
| **AE** (autoencoder) | Deterministic projection | None |
| **VAE** (variational) | Gaussian (mu, logvar) + reparameterization | KL divergence |
| **VQ-VAE** (vector quantized) | Discrete codebook (512 entries) | Commitment loss + EMA updates |

All three share the same encoder (4-layer transformer with Time2Vec positional encoding and CLS token) and decoder (2-layer transformer with learned positional queries).

### Input format

Each alert is a variable-length photometry sequence `(L, 7)`:

| Channel | Description |
|---|---|
| `log1p(dt)` | Log time since first detection (days) |
| `log1p(dt_prev)` | Log inter-observation gap (days) |
| `logflux` | Log10 flux |
| `logflux_err` | Log flux uncertainty |
| `band_g` | One-hot: ZTF g-band |
| `band_r` | One-hot: ZTF r-band |
| `band_i` | One-hot: ZTF i-band |

Sequences are padded to max length 257 and normalized using global training set statistics.

## Results

### Architecture comparison (10 latent dimensions, 200 epochs each)

#### Coarse 5-class linear probe accuracy

| Dim | Bytes/alert | Compression | AE | VAE | VQ-VAE |
|-----|-------------|-------------|-----|-----|--------|
| 2 | 8B | 900x | **76.4%** | 73.7% | 75.8% |
| 4 | 16B | 450x | **79.4%** | 73.4% | 78.1% |
| 8 | 32B | 225x | **82.6%** | 77.0% | 77.4% |
| 16 | 64B | 112x | **84.6%** | 79.1% | 78.8% |
| 32 | 128B | 56x | **85.8%** | 82.3% | 78.5% |
| 64 | 256B | 28x | **86.5%** | 84.3% | 78.7% |
| 128 | 512B | 14x | **87.1%** | 86.3% | 80.1% |
| 256 | 1024B | 7x | **88.4%** | 86.8% | 80.0% |
| 512 | 2048B | 3.5x | **88.2%** | 87.1% | 80.7% |
| 1024 | 4096B | 1.8x | **88.5%** | 86.0% | 79.9% |

#### Continuous reconstruction MSE (lower is better)

| Dim | AE | VAE | VQ-VAE |
|-----|-----|-----|--------|
| 2 | **0.38** | 0.51 | 0.38 |
| 8 | **0.29** | 0.63 | 0.39 |
| 32 | **0.27** | 0.63 | 0.39 |
| 64 | **0.26** | 0.64 | 0.38 |
| 256 | **0.26** | 0.63 | 0.38 |
| 1024 | **0.26** | 0.62 | 0.38 |

#### Band reconstruction accuracy (3-class: g/r/i)

| Dim | AE | VAE | VQ-VAE |
|-----|-----|-----|--------|
| 2 | 55.3% | 52.7% | 56.1% |
| 64 | 62.7% | 57.1% | 56.2% |
| 256 | **63.0%** | 57.6% | 56.3% |
| 1024 | **63.2%** | 57.4% | 56.5% |

### Key findings

1. **The plain autoencoder (AE) wins across the board** — better reconstruction AND better downstream classification at every latent dimension. The KL penalty in the VAE hurts both metrics without compensating benefit for compression.

2. **Classification saturates at dim ~256-512** (~88.5% coarse accuracy). Reconstruction saturates earlier at dim ~64 (MSE ~0.26).

3. **VQ-VAE plateaus at ~80%** regardless of latent dimension. The 512-entry codebook is the true bottleneck — all dims map to ~95 active codes. A multi-code or hierarchical VQ approach would be needed to improve this.

4. **The sweet spot is AE dim=64**: 86.5% coarse accuracy at 28x compression (256 bytes/alert). Going to dim=256 gains 2pp at 4x more bytes.

5. **Context vs the full XGBoost pipeline**: the AE at dim=256 (88.4%) approaches the XGBoost catalog-only baseline (65.3%) + fitting features (94.4%) using only raw photometry, no engineered features or catalog cross-matches. Adding alert metadata and catalog features to the encoder input is expected to close much of this gap.

### Classes

**Coarse (5-class):** SNIa, SNcc, Cataclysmic, AGN, TDE

**Fine (10-class):** SN Ia, SN Ib, SN Ic, SN II, SN IIP, SN IIn, SN IIb, Cataclysmic, AGN, TDE

### Data

- 18,245 labeled ZTF transients from the AppleCiDEr sample
- Train: 12,771 / Val: 2,737 / Test: 2,737
- Photometry stored as NPZ files with 7-channel event sequences

## Usage

### Training a single model

```bash
cd src

# AE with 64-dimensional latent space
python train.py \
    --data-dir /path/to/photo_events \
    --output-dir ../runs \
    --mode ae \
    --latent-dims 64 \
    --epochs 200

# VAE
python train.py --mode vae --latent-dims 64 --beta 1.0 ...

# VQ-VAE with 512-entry codebook
python train.py --mode vqvae --latent-dims 64 --num-codes 512 ...
```

### Latent dimension sweep

```bash
python train.py \
    --data-dir /path/to/photo_events \
    --output-dir ../runs \
    --mode ae \
    --latent-dims 2 4 8 16 32 64 128 256 512 1024 \
    --epochs 200
```

### Linear probe evaluation

```bash
python linear_probe.py \
    --runs-dir ../runs \
    --output-dir ../analysis
```

### SLURM (OzSTAR)

```bash
# Full AE vs VAE vs VQ-VAE sweep (30 models, ~8 hours on A100)
sbatch slurm/sweep_all_modes.sh
```

## Architecture

```
Encoder:
  Linear(7, 128) + Time2Vec(128) + CLS token
  TransformerEncoder(4 layers, 8 heads, 512 FFN, dropout=0.3)
  CLS token → LayerNorm → bottleneck

Bottleneck (mode-dependent):
  AE:    Linear(128, latent_dim)
  VAE:   Linear(128, latent_dim) × 2 → (mu, logvar) → reparameterize
  VQ-VAE: Linear(128, latent_dim) → VectorQuantizer(512 codes, EMA)

Decoder:
  Linear(latent_dim, 128) → broadcast to 257 positions + learned pos embed
  TransformerEncoder(2 layers, 8 heads, 512 FFN, dropout=0.2)
  head_cont: Linear(128, 4)  → continuous channels (MSE loss)
  head_band: Linear(128, 3)  → band logits (CE loss)
```

Parameters: ~1.2M (dim=2) to ~1.6M (dim=1024).

## File structure

```
alert-compression/
├── src/
│   ├── model.py          # LightCurveCompressor (AE/VAE/VQ-VAE)
│   ├── dataset.py         # PhotoNPZDataset + collate_fn
│   ├── train.py           # Training loop + latent dim sweep
│   └── linear_probe.py    # Downstream classification evaluation
├── slurm/
│   └── sweep_all_modes.sh # Full comparison job
├── runs/                  # Trained models + embeddings (per run)
└── analysis/              # Linear probe results + plots
```
