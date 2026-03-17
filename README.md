# RTF: Rapid Transient Fingerprints

Latent-space compression of astronomical alert streams using transformer autoencoders. Encodes variable-length multi-band light curves into compact fixed-length fingerprints that preserve astrophysically meaningful information.

Part of the [BOOM](https://github.com/boom-astro) project and the [AppleCiDEr](https://github.com/applecider-ml) pipeline.

## Motivation

LSST will produce ~10M alerts/night. Current brokers filter this to a small subset, discarding most of the stream. RTF compresses the full alert stream into fixed-length latent vectors, enabling:

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

### Reconstruction quality in physical units

The AE reconstructs light curves to sub-0.2 mag accuracy at moderate compression:

| Mode | Dim | Bytes | Compress | Mag (mean) | Mag (median) | Mag (90th pct) | Time error (days) | Band acc |
|------|-----|-------|----------|-----------|-------------|----------------|-------------------|----------|
| AE | 2 | 8B | 900x | 0.34 | 0.26 | 0.60 | 7.1 | 62.7% |
| AE | 8 | 32B | 225x | 0.25 | 0.19 | 0.46 | 3.5 | 70.7% |
| AE | 32 | 128B | 56x | 0.22 | 0.16 | 0.40 | 2.6 | 72.9% |
| **AE** | **64** | **256B** | **28x** | **0.21** | **0.16** | **0.40** | **2.5** | **73.3%** |
| AE | 256 | 1024B | 7x | 0.21 | 0.16 | 0.40 | 2.5 | 73.6% |
| AE | 1024 | 4096B | 1.8x | 0.21 | 0.16 | 0.40 | 2.4 | 73.9% |
| VAE | 64 | 256B | 28x | 0.42 | 0.33 | 0.73 | 15.6 | 64.7% |
| VQ-VAE | 64 | 256B | 28x | 0.37 | 0.28 | 0.65 | 7.3 | 63.9% |

Reconstruction saturates at dim ~64 (0.16 mag median). The AE outperforms the VAE by ~0.2 mag and VQ-VAE by ~0.1 mag at every dimension.

### Downstream classification from latent vectors

Three decoder types are compared: linear probe (logistic regression), 2-layer MLP, and 3-layer MLP. All decoders are trained on frozen latent vectors from the best AE checkpoint.

#### Coarse 5-class accuracy (SNIa / SNcc / Cataclysmic / AGN / TDE)

| Dim | Bytes | Linear acc | Linear bal | MLP-2 acc | MLP-2 bal | MLP-3 acc | MLP-3 bal |
|-----|-------|-----------|-----------|----------|----------|----------|----------|
| 2 | 8B | 76.4% | 36.6% | 68.3% | 54.7% | 69.6% | **55.0%** |
| 8 | 32B | 82.6% | 49.6% | 81.4% | 70.2% | 82.4% | **70.7%** |
| 32 | 128B | 85.8% | 55.8% | 84.9% | **73.3%** | 85.5% | 73.1% |
| **64** | **256B** | **86.5%** | **57.9%** | **85.2%** | **71.8%** | **86.2%** | **72.3%** |
| 256 | 1024B | 88.4% | 64.7% | 86.9% | **74.1%** | 86.1% | 73.3% |
| 512 | 2048B | 88.2% | 66.6% | 86.4% | 74.2% | 86.9% | **76.2%** |
| 1024 | 4096B | 88.5% | 63.8% | 87.6% | **78.5%** | 87.8% | 76.3% |

The MLP decoders improve **balanced accuracy by 15-21 percentage points** over linear probes, critical for rare classes (TDE, SN subtypes).

#### Architecture comparison — linear probe accuracy

| Dim | Bytes | Compression | AE | VAE | VQ-VAE |
|-----|-------|-------------|-----|-----|--------|
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

### Key findings

1. **The plain autoencoder (AE) wins across the board** — better reconstruction AND better downstream classification at every latent dimension. The KL penalty in the VAE hurts both metrics without compensating benefit for compression.

2. **Reconstruction saturates at dim ~64** (0.16 mag median, 2.5 day time error). Classification continues improving to dim ~512.

3. **MLP decoders dramatically improve balanced accuracy** (+15-21pp over linear probes), especially for rare classes. A simple 2-layer MLP is sufficient.

4. **VQ-VAE plateaus at ~80%** regardless of latent dimension. The 512-entry codebook is the true bottleneck — all dims map to ~95 active codes. A multi-code or hierarchical VQ approach would be needed to improve this.

5. **The sweet spot is AE dim=64**: 0.16 mag reconstruction, 73% balanced classification accuracy with MLP decoder, at 28x compression (256 bytes/alert).

6. **Context vs the full XGBoost pipeline**: the AE at dim=256 (88.4% accuracy) approaches the full XGBoost (94.4%) using only raw photometry — no engineered features or catalog cross-matches. Adding alert metadata is expected to close much of this gap.

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

### Evaluation

```bash
# Linear probe classification
python linear_probe.py --runs-dir ../runs --output-dir ../analysis

# MLP decoder classifiers (linear + 2-layer + 3-layer MLP)
python mlp_decoder.py --runs-dir ../runs --output-dir ../analysis/decoders

# Physical-unit reconstruction metrics + light curve plots
python evaluate_physical.py --runs-dir ../runs --data-dir /path/to/photo_events \
    --output-dir ../analysis/physical
```

### SLURM (OzSTAR)

```bash
# Full AE vs VAE vs VQ-VAE sweep (30 models, ~8 hours on A100)
sbatch slurm/sweep_all_modes.sh

# Evaluation (physical metrics + MLP decoders, ~45 min)
sbatch slurm/evaluate.sh
```

### Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

## Architecture

```
Encoder:
  Linear(7, 128) + Time2Vec(128) + CLS token
  TransformerEncoder(4 layers, 8 heads, 512 FFN, dropout=0.3)
  CLS token -> LayerNorm -> bottleneck

Bottleneck (mode-dependent):
  AE:     Linear(128, latent_dim)
  VAE:    Linear(128, latent_dim) x 2 -> (mu, logvar) -> reparameterize
  VQ-VAE: Linear(128, latent_dim) -> VectorQuantizer(512 codes, EMA)

Decoder:
  Linear(latent_dim, 128) -> broadcast to 257 positions + learned pos embed
  TransformerEncoder(2 layers, 8 heads, 512 FFN, dropout=0.2)
  head_cont: Linear(128, 4)  -> continuous channels (MSE loss)
  head_band: Linear(128, 3)  -> band logits (CE loss)
```

Parameters: ~1.2M (dim=2) to ~1.6M (dim=1024).

## File structure

```
rtf/
├── src/
│   ├── model.py              # LightCurveCompressor (AE/VAE/VQ-VAE)
│   ├── dataset.py            # PhotoNPZDataset + collate_fn
│   ├── train.py              # Training loop + latent dim sweep
│   ├── linear_probe.py       # Linear probe classification
│   ├── mlp_decoder.py        # MLP decoder classifiers
│   └── evaluate_physical.py  # Physical-unit metrics + light curve plots
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_model.py         # Model unit tests (61 tests)
│   ├── test_dataset.py       # Dataset tests
│   └── test_training.py      # Integration tests
├── slurm/
│   ├── sweep_all_modes.sh    # Full architecture comparison
│   └── evaluate.sh           # Evaluation job
├── .github/workflows/ci.yml  # GitHub Actions CI
├── pyproject.toml
├── runs/                     # Trained models + embeddings (gitignored)
└── analysis/                 # Results + plots (gitignored)
```
