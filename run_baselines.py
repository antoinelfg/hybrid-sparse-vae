"""Baseline comparison for Hybrid Sparse VAE.

Trains 4 baselines on the same toy sinusoid data and produces a comparison
table against the Hybrid Sparse VAE checkpoint.

Baselines:
  1. Vanilla AE        — direct MLP encode/decode, no latent structure
  2. Vanilla VAE        — Gaussian z with reparameterization
  3. Sparse AE (L1)     — AE + L1 penalty on activations
  4. OMP (oracle)       — Orthogonal Matching Pursuit with DCT dictionary (no learning)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Shared data generator (same as train.py)
# ---------------------------------------------------------------------------

def make_toy_data(
    n_samples: int = 2000,
    length: int = 128,
    n_components: int = 3,
    seed: int = 0,
) -> TensorDataset:
    rng = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 2 * torch.pi, length)
    signals = []
    for _ in range(n_samples):
        freqs = torch.randint(1, 20, (n_components,), generator=rng).float()
        amps = torch.rand(n_components, generator=rng) + 0.3
        phases = torch.rand(n_components, generator=rng) * 2 * torch.pi
        sig = sum(a * torch.sin(f * t + p) for a, f, p in zip(amps, freqs, phases))
        signals.append(sig)
    X = torch.stack(signals).unsqueeze(1) / 4.0
    return TensorDataset(X)


# ---------------------------------------------------------------------------
#  Shared encoder/decoder (same architecture as HybridSparseVAE)
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, hidden: int = 256, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, output_dim),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.view(x.size(0), -1))


class SharedDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden: int = 256, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, output_dim),
        )
    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).unsqueeze(1)  # [B, 1, T]


# ===========================================================================
#  Baseline 1: Vanilla Autoencoder
# ===========================================================================

class VanillaAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = SharedEncoder(output_dim=latent_dim)
        self.decoder = SharedDecoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), {"z": z}


# ===========================================================================
#  Baseline 2: Vanilla VAE (Gaussian)
# ===========================================================================

class VanillaVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = SharedEncoder(output_dim=latent_dim * 2)  # μ + logσ²
        self.decoder = SharedDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return self.decoder(z), {"z": z, "mu": mu, "logvar": logvar, "kl": kl}


# ===========================================================================
#  Baseline 3: Sparse AE (L1)
# ===========================================================================

class SparseAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = SharedEncoder(output_dim=latent_dim)
        self.decoder = SharedDecoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        l1 = z.abs().sum(dim=-1).mean()
        sparsity = (z.abs() < 0.01).float().mean().item()
        return self.decoder(z), {"z": z, "l1": l1, "sparsity": sparsity}


# ===========================================================================
#  Baseline 4: Orthogonal Matching Pursuit (oracle)
# ===========================================================================

def build_dct_dictionary(n_atoms: int, signal_length: int) -> Tensor:
    """Build overcomplete DCT dictionary [signal_length, n_atoms]."""
    D = torch.zeros(signal_length, n_atoms)
    for k in range(n_atoms):
        if k == 0:
            D[:, k] = 1.0 / math.sqrt(signal_length)
        else:
            n = torch.arange(signal_length, dtype=torch.float32)
            D[:, k] = math.sqrt(2.0 / signal_length) * torch.cos(
                math.pi * k * (2 * n + 1) / (2 * signal_length)
            )
    return D


def omp(signal: Tensor, dictionary: Tensor, n_nonzero: int) -> Tensor:
    """Orthogonal Matching Pursuit for a single signal.

    Args:
        signal: [T] target signal
        dictionary: [T, n_atoms] dictionary matrix
        n_nonzero: number of atoms to select

    Returns:
        coeffs: [n_atoms] sparse coefficient vector
    """
    T, K = dictionary.shape
    residual = signal.clone()
    selected = []
    coeffs = torch.zeros(K, device=signal.device)

    for _ in range(n_nonzero):
        # Find atom most correlated with residual
        correlations = dictionary.T @ residual  # [K]
        # Exclude already selected atoms
        for idx in selected:
            correlations[idx] = 0.0
        best = correlations.abs().argmax().item()
        selected.append(best)

        # Solve least squares on selected atoms
        D_sel = dictionary[:, selected]  # [T, |selected|]
        # x = (D_sel^T D_sel)^{-1} D_sel^T signal
        c = torch.linalg.lstsq(D_sel, signal).solution  # [|selected|]
        residual = signal - D_sel @ c

    for i, idx in enumerate(selected):
        coeffs[idx] = c[i]

    return coeffs


def run_omp_baseline(X: Tensor, n_atoms: int = 128, n_nonzero: int = 33) -> dict:
    """Run OMP on all signals and return metrics.

    Args:
        X: [N, 1, T] signals
        n_atoms: dictionary size
        n_nonzero: target sparsity (number of active atoms)
    """
    N, _, T = X.shape
    D = build_dct_dictionary(n_atoms, T).to(X.device)

    total_mse = 0.0
    total_sparsity = 0.0

    for i in range(N):
        sig = X[i, 0]  # [T]
        coeffs = omp(sig, D, n_nonzero)
        recon = D @ coeffs
        mse = F.mse_loss(recon, sig, reduction='sum').item()
        total_mse += mse
        total_sparsity += (coeffs == 0).float().mean().item()

    return {
        "recon_mse": total_mse / N,
        "sparsity": total_sparsity / N,
        "n_active": n_nonzero,
    }


# ===========================================================================
#  Training function (shared for all learned baselines)
# ===========================================================================

def train_baseline(
    model: nn.Module,
    data: TensorDataset,
    name: str,
    epochs: int = 2000,
    lr: float = 3e-4,
    batch_size: int = 64,
    beta: float = 1.0,
    lambda_l1: float = 0.01,
    device: str = "cuda",
) -> dict:
    """Train a baseline model and return final metrics."""
    model = model.to(device)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_recon = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_recon = 0.0
        epoch_reg = 0.0

        # Cosine LR decay in second half
        if epoch > epochs // 2:
            frac = (epoch - epochs // 2) / (epochs // 2)
            lr_now = lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * frac))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_recon, info = model(batch_x)
            recon_loss = F.mse_loss(x_recon, batch_x, reduction='sum') / batch_x.size(0)

            # Regularization
            if name == "vae" and "kl" in info:
                reg = beta * info["kl"]
            elif name == "sparse_ae" and "l1" in info:
                reg = lambda_l1 * info["l1"]
            else:
                reg = torch.tensor(0.0, device=device)

            loss = recon_loss + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_reg += reg.item() if isinstance(reg, Tensor) else reg

        n_batches = len(loader)
        avg_recon = epoch_recon / n_batches
        avg_reg = epoch_reg / n_batches
        best_recon = min(best_recon, avg_recon)

        if epoch % 200 == 0 or epoch == 1:
            log.info(
                f"  [{name:>10s}] Epoch {epoch:4d} | "
                f"recon {avg_recon:.4f} | reg {avg_reg:.4f} | "
                f"best {best_recon:.4f}"
            )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_x = data.tensors[0].to(device)
        x_recon, info = model(all_x)
        final_mse = F.mse_loss(x_recon, all_x, reduction='sum').item() / all_x.size(0)

        # Compute sparsity for models with z
        z = info.get("z", None)
        sparsity = (z.abs() < 0.01).float().mean().item() if z is not None else 0.0

    return {
        "recon_mse": final_mse,
        "best_recon": best_recon,
        "sparsity": sparsity,
    }


# ===========================================================================
#  Main comparison
# ===========================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Generate data
    data = make_toy_data(n_samples=2000, length=128, n_components=3, seed=0)
    X = data.tensors[0]
    log.info(f"Data: {X.shape}, range [{X.min():.3f}, {X.max():.3f}]")

    # Zero-prediction baseline
    zero_mse = F.mse_loss(torch.zeros_like(X), X, reduction='sum').item() / X.size(0)
    log.info(f"Zero-prediction MSE: {zero_mse:.4f}")

    results = {"zero_pred": {"recon_mse": zero_mse, "best_recon": zero_mse, "sparsity": 1.0}}

    # --- Baseline 1: Vanilla AE ---
    log.info("\n=== Vanilla AE ===")
    ae = VanillaAE(latent_dim=128)
    results["vanilla_ae"] = train_baseline(ae, data, "ae", epochs=2000, device=device)

    # --- Baseline 2: Vanilla VAE ---
    log.info("\n=== Vanilla VAE (β=0.005) ===")
    vae = VanillaVAE(latent_dim=128)
    results["vanilla_vae"] = train_baseline(
        vae, data, "vae", epochs=2000, beta=0.005, device=device
    )

    # --- Baseline 3: Sparse AE (L1) ---
    log.info("\n=== Sparse AE (λ=0.01) ===")
    sae = SparseAE(latent_dim=128)
    results["sparse_ae"] = train_baseline(
        sae, data, "sparse_ae", epochs=2000, lambda_l1=0.01, device=device
    )

    # --- Baseline 4: OMP (oracle, no learning) ---
    log.info("\n=== OMP (oracle, 33 atoms / 128) ===")
    t0 = time.time()
    # Match our model's observed sparsity: ~33/128 active atoms
    omp_results = run_omp_baseline(X, n_atoms=128, n_nonzero=33)
    log.info(
        f"  [       OMP] recon {omp_results['recon_mse']:.4f} | "
        f"sparsity {omp_results['sparsity']:.2%} | "
        f"time {time.time() - t0:.1f}s"
    )
    results["omp_33"] = {
        "recon_mse": omp_results["recon_mse"],
        "best_recon": omp_results["recon_mse"],
        "sparsity": omp_results["sparsity"],
    }

    # OMP with fewer atoms (sparser)
    log.info("\n=== OMP (oracle, 10 atoms / 128) ===")
    t0 = time.time()
    omp_sparse = run_omp_baseline(X, n_atoms=128, n_nonzero=10)
    log.info(
        f"  [  OMP (10)] recon {omp_sparse['recon_mse']:.4f} | "
        f"sparsity {omp_sparse['sparsity']:.2%} | "
        f"time {time.time() - t0:.1f}s"
    )
    results["omp_10"] = {
        "recon_mse": omp_sparse["recon_mse"],
        "best_recon": omp_sparse["recon_mse"],
        "sparsity": omp_sparse["sparsity"],
    }

    # --- Summary ---
    log.info("\n" + "=" * 80)
    log.info("  BASELINE COMPARISON RESULTS")
    log.info("=" * 80)
    log.info(f"{'Model':<20s} {'Recon MSE':>12s} {'Best MSE':>12s} {'Sparsity':>10s}")
    log.info("-" * 56)
    for name, r in results.items():
        log.info(
            f"{name:<20s} {r['recon_mse']:>12.4f} {r['best_recon']:>12.4f} "
            f"{r['sparsity']:>9.2%}"
        )
    log.info("-" * 56)
    log.info("NOTE: Compare Hybrid Sparse VAE best recon (~3.9) against these.")
    log.info("      Our model adds: structured sparsity, generative sampling,")
    log.info("      interpretable polar decomposition (sign × magnitude × dictionary).")


if __name__ == "__main__":
    main()
