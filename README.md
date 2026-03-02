# Hybrid Sparse VAE

**Variational Auto-Encoder with Structured Sparsity via Polar Factorization**

$$z = A \cdot (\gamma \odot \delta)$$

where $\gamma \sim \text{Gamma}(k, \theta)$ with $k < 1$ (super-Gaussian sparsity) and $\delta \in \{-1, 0, +1\}$ (ternary structure via Gumbel-Softmax).

---

## Key Contributions

1. **Implicit Reparameterization Gradients (IRG)** for $\text{Gamma}(k < 1)$ — numerically stable via log-space Geddes series
2. **Polar factorization** decoupling magnitude (continuous) from structure (discrete)
3. **Exact KL divergences** — no Monte-Carlo estimation of regularization terms
4. **4-Phase "Rocket Launch" training** — progressive curriculum: soft→stochastic→KL ramp→convergence

## 🏆 Key Results & Baselines (Toy Sinusoids)

We accept a reconstruction cost vs. iterative matching pursuit (OMP) in exchange for **one-shot amortized inference**, **generative sampling**, and **interpretable polar decomposition** — while outperforming all comparable structured generative baselines at iso-sparsity.

| Model | Recon MSE ↓ | Sparsity† | Generative | Structured | Notes |
|-------|:-----------:|:---------:|:----------:|:----------:|:------|
| Vanilla AE | **0.000** | 2% | ✗ | ✗ | Pure compression |
| OMP (10 atoms) | **0.354** | 92% | ✗ | ✗ | Iterative, non-amortized |
| Vanilla VAE (β=0.005) | **0.069** | 1% | ✅ | ✗ | Non-sparse generative |
| Spike-Slab VAE (β=0.05) | **3.01** | ~68%* | ✅ | ✅ | *Sparsity drops during training |
| **Hybrid VAE (DCT)** | **1.82** | **68%** | **✅** | **✅** | Robust, deterministic init |
| **Hybrid VAE (Random init)** | **1.08** (median) | **68%** | **✅** | **✅** | **3× better than Spike-Slab** |
| β-VAE / VQ-VAE | ~8.69 | 0-4% | ⚠️ | ✗ | Complete codebook/posterior collapse |

> **†** Sparsity measured at training time (Gumbel noise active). At iso-sparsity (~68%), our model outperforms the Spike-and-Slab VAE by **3×**.


## Repository Structure

```
hybrid-sparse-vae/
├── train.py                # Hydra training script (4-phase schedule)
├── models/                 # Full HybridSparseVAE model
├── modules/                # Dictionary matrix, Latent Space (polar factorization)
├── math_ops/               # Core math: IRG Gamma sampler, hypergeometric series
├── utils/                  # Objectives (exact KL), visualization
├── tests/                  # Unit & smoke tests
├── scripts/
│   ├── slurm/              # SLURM batch scripts for cluster runs
│   └── baselines/          # Baseline comparison scripts (AE, VAE, OMP, Spike-Slab, …)
├── results/                # Experiment outputs organized by run (not tracked)
└── data/                   # Generated at runtime (not tracked)
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run gradient check (validates IRG correctness)
python -m pytest tests/ -v

# Train on toy data (sum of sinusoids) — champion config (DCT)
python train.py epochs=3000 dict_init=dct

# With learned random dictionary (evaluates multi-seed)
python train.py epochs=3000 dict_init=random dict_lr_mult=0.1

# Run MNIST campaign (SLURM array)
sbatch scripts/slurm/launch_mnist.sh

# Vis & Eval MNIST High-K Models
python scripts/vis_mnist_atoms.py --checkpoint results/mnist_campaign/atoms_256_seed_42/hybrid_vae_final.pt --n-atoms 256
python scripts/evaluate_generation.py --checkpoint results/mnist_campaign/atoms_256_seed_42/hybrid_vae_final.pt --dataset mnist --n-atoms 256 --input-length 784
```

## Training Phases ("Rocket Launch" Schedule)

```
Phase 1 [ep 1-400]     soft sampling, β=0         → sculpt topology
Phase 2 [ep 401-500]   stochastic, β=0            → immunize to noise
Phase 3 [ep 501-1000]  stochastic, β ramp 0→max   → gradual KL & sparsification
Phase 4 [ep 1001-3000] stochastic, β=max          → cosine LR decay to convergence
```

## 🚀 Experimental Campaigns & Discoveries

### 1. MNIST "Cold Start" Campaign
High-dimensional dense images (MNIST) impose a strict **"Noise Floor"** on the Gamma shape parameter $k$. When $k < 1$, the multiplicative noise ($1/\sqrt{k}$) exceeds 100%, causing catastrophic variance explosion and posterior collapse. By enforcing a stability constraint ($k_{min}=10$) and skipping the deterministic warmup ("Cold Start"), we decouple sparsity (handled strictly by ternary $\delta$) from magnitude noise:
* **Generative Diversity (Recall):** 94.05% vs 83.40% (Gaussian baseline)
* **Fidelity (MMD):** 0.0506 (3x better than Gaussian baseline)
* **Learned Atoms:** Sharp, stroke-like parts-based decomposition, avoiding the holistic "fuzzy blobs" of standard Gaussian VAEs.

### 2. Linear Decoder Validation (Fourier Transform)
With a strictly linear decoder ($x = A \cdot z$) and a gentle spectral warmup schedule, the model try to re-invent the 1D Fourier Transform on sinusoidal data, learning distinct pure sinusoidal basis functions.

### 3. Stability & Architecture Ablations
* **Dictionary Init:** Random initialization reaches the best absolute MSE (1.08), but DCT initialization provides robust, deterministic convergence (1.82). Freezing the dictionary during phase 1/2 and unfreezing it later is optimal for DCT.
* **Component Ablations:** Removing the phase schedule or dictionary normalization causes catastrophic failure (MSE > 7.0). Switching to a Gaussian magnitude prior improves reconstruction but completely destroys the sparsity profile, pushing $k \to 0$.
* **Generative Quality Metrics:** Integrated W&B and comprehensive generative evaluation suites measuring Maximum Mean Discrepancy (MMD), Precision & Recall, and Power Spectral Density (PSD) error to formally assess generative capabilities vs. standard VAEs.

## Theory

The latent code is factored as $z = A \cdot B$ where $B_i = \gamma_i \cdot \delta_i$:

- $\gamma_i \sim \text{Gamma}(k_i, \theta_i)$ — magnitude (continuous, sparse when $k < 1$)
- $\delta_i \in \{-1, 0, +1\}$ — structure (discrete, Gumbel-Softmax ST relaxation)
- $A \in \mathbb{R}^{d \times N}$ — dictionary (DCT or learned) with column-normalization

Gradients flow through $\gamma$ via **Implicit Reparameterization**:

$$\frac{dz}{dk} = -\frac{\partial P(k, z/\theta) / \partial k}{p(z; k, \theta)}$$

## Citation

*Coming soon.*
