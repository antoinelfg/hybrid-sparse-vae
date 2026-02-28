# Hybrid Sparse VAE

**Variational Auto-Encoder with Structured Sparsity via Polar Factorization**

$$z = A \cdot (\gamma \odot \delta)$$

where $\gamma \sim \text{Gamma}(k, \theta)$ with $k < 1$ (super-Gaussian sparsity) and $\delta \in \{-1, 0, +1\}$ (ternary structure via Gumbel-Softmax).

---

## Key Contributions

1. **Implicit Reparameterization Gradients (IRG)** for $\text{Gamma}(k < 1)$ — numerically stable via log-space Geddes series expansion
2. **Polar factorization** decoupling magnitude (continuous) from structure (discrete)
3. **Exact KL divergences** — no Monte-Carlo estimation of regularization terms

## Repository Structure

```
hybrid-sparse-vae/
├── math_ops/           # Core math: IRG Gamma sampler, hypergeometric series
├── modules/            # Neural layers, Dictionary matrix A, Latent Space
├── models/             # Full HybridSparseVAE model
├── utils/              # Objectives (exact KL), visualization
├── tests/              # Unit & smoke tests
└── train.py            # Hydra training script
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run gradient check (validates IRG correctness)
python -m pytest tests/test_implicit_gamma.py -v

# Train on toy data (sum of sinusoids)
python train.py epochs=100 n_atoms=32

# Override any config via CLI
python train.py lr=5e-4 beta_gamma=0.5 latent_dim=64
```

## Theory

The latent code $z$ is factored as:

$$z = A \cdot B, \quad B_i = \gamma_i \cdot \delta_i$$

- $\gamma_i \sim \text{Gamma}(k_i, \theta_i)$ — magnitude (continuous, sparse when $k < 1$)
- $\delta_i \in \{-1, 0, +1\}$ — structure (discrete, Gumbel-Softmax relaxation)
- $A \in \mathbb{R}^{d \times N}$ — learnable dictionary with column-normalization

Gradients flow through $\gamma$ via **Implicit Reparameterization**:

$$\frac{dz}{dk} = -\frac{\partial F / \partial k}{\partial F / \partial z} = -\frac{\partial P(k, z/\theta) / \partial k}{p(z; k, \theta)}$$

where $\partial P / \partial k$ is computed via the Geddes series in log-space for numerical stability.

## Citation

*Coming soon.*
