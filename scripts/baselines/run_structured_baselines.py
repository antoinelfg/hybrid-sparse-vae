"""Structured baselines for Hybrid Sparse VAE comparison.

These are "honestly structured" generative models that pay a reconstruction
cost for structure/sparsity — the fair comparison for our model.

Baselines:
  1. β-VAE (β=4,10,20)     — disentanglement via KL pressure
  2. VQ-VAE (small book)   — discrete latent codes, same #atoms
  3. Spike-and-Slab VAE    — Bernoulli gating × Gaussian magnitude
  4. SC-VAE (Laplace)      — sparse coding VAE with Laplace prior
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
#  Shared
# ---------------------------------------------------------------------------

def make_toy_data(n_samples=2000, length=128, n_components=3, seed=0):
    rng = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 2 * torch.pi, length)
    signals = []
    for _ in range(n_samples):
        freqs = torch.randint(1, 20, (n_components,), generator=rng).float()
        amps = torch.rand(n_components, generator=rng) + 0.3
        phases = torch.rand(n_components, generator=rng) * 2 * torch.pi
        sig = sum(a * torch.sin(f * t + p) for a, f, p in zip(amps, freqs, phases))
        signals.append(sig)
    return TensorDataset(torch.stack(signals).unsqueeze(1) / 4.0)


class Enc(nn.Module):
    """Shared MLP encoder [B,1,128] → [B,D]."""
    def __init__(self, out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True),
            nn.Linear(256, out),
        )
    def forward(self, x): return self.net(x.view(x.size(0), -1))


class Dec(nn.Module):
    """Shared MLP decoder [B,D] → [B,1,128]."""
    def __init__(self, inp=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 256), nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True),
            nn.Linear(256, 128),
        )
    def forward(self, z): return self.net(z).unsqueeze(1)


# ===========================================================================
#  1. β-VAE  (Gaussian, high β)
# ===========================================================================

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Enc(out=latent_dim * 2)
        self.decoder = Dec(inp=latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        return self.decoder(z), {"z": z, "kl": kl}


# ===========================================================================
#  2. VQ-VAE  (Vector Quantization)
# ===========================================================================

class VectorQuantizer(nn.Module):
    """Straight-through VQ with EMA codebook update."""
    def __init__(self, n_embeddings=128, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_embeddings, 1.0 / n_embeddings)

    def forward(self, z_e):
        # z_e: [B, D]
        # Distances to codebook entries
        dists = (z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)  # [B, K]
        indices = dists.argmin(dim=-1)  # [B]
        z_q = self.codebook(indices)  # [B, D]

        # Losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())  # move codebook to encoder
        commitment_loss = F.mse_loss(z_e, z_q.detach())  # move encoder to codebook

        # Straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        # Perplexity (codebook usage)
        one_hot = F.one_hot(indices, self.n_embeddings).float()
        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        loss = codebook_loss + self.commitment_cost * commitment_loss
        return z_q_st, {"vq_loss": loss, "perplexity": perplexity, "indices": indices}


class VQVAE(nn.Module):
    def __init__(self, n_embeddings=128, embedding_dim=128):
        super().__init__()
        self.encoder = Enc(out=embedding_dim)
        self.vq = VectorQuantizer(n_embeddings, embedding_dim)
        self.decoder = Dec(inp=embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_info = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_info


# ===========================================================================
#  3. Spike-and-Slab VAE (Bernoulli × Gaussian)
# ===========================================================================

class SpikeSlabVAE(nn.Module):
    """Spike-and-Slab VAE: z = mask ⊙ gamma, mask ~ Bernoulli(π), gamma ~ N(μ,σ²).

    Uses Gumbel-Sigmoid for differentiable Bernoulli and reparameterization
    for Gaussian.
    """
    def __init__(self, latent_dim=128, temp=0.5):
        super().__init__()
        self.encoder = Enc(out=latent_dim * 3)  # μ, logσ², logit(π)
        self.decoder = Dec(inp=latent_dim)
        self.latent_dim = latent_dim
        self.temp = temp

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar, logit_pi = torch.split(h, self.latent_dim, dim=-1)

        # Gaussian component (slab)
        std = (0.5 * logvar).exp()
        gaussian_sample = mu + std * torch.randn_like(std)

        # Bernoulli gate (spike) — Gumbel-Sigmoid
        if self.training:
            u = torch.rand_like(logit_pi).clamp(1e-6, 1 - 1e-6)
            gumbel = torch.log(u) - torch.log(1 - u)
            mask_soft = torch.sigmoid((logit_pi + gumbel) / self.temp)
            # Straight-through
            mask_hard = (mask_soft > 0.5).float()
            mask = mask_hard - mask_soft.detach() + mask_soft
        else:
            mask = (logit_pi > 0).float()

        z = mask * gaussian_sample

        # KL for Gaussian part (only on active dims)
        kl_gauss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        # KL for Bernoulli gate
        pi = torch.sigmoid(logit_pi)
        prior_pi = 0.3  # sparse prior: 30% active
        kl_bernoulli = (
            pi * (torch.log(pi + 1e-8) - math.log(prior_pi))
            + (1 - pi) * (torch.log(1 - pi + 1e-8) - math.log(1 - prior_pi))
        ).sum(-1).mean()

        sparsity = (mask < 0.5).float().mean().item()

        return self.decoder(z), {
            "z": z, "mask": mask,
            "kl_gauss": kl_gauss, "kl_bernoulli": kl_bernoulli,
            "sparsity": sparsity,
        }


# ===========================================================================
#  4. SC-VAE (Sparse Coding VAE with Laplace prior)
# ===========================================================================

class SCVAE(nn.Module):
    """Sparse Coding VAE: Gaussian encoder + Laplace prior.

    KL(N(μ,σ²) || Laplace(0,b)) is computed analytically.
    """
    def __init__(self, latent_dim=128, prior_scale=1.0):
        super().__init__()
        self.encoder = Enc(out=latent_dim * 2)
        self.decoder = Dec(inp=latent_dim)
        self.latent_dim = latent_dim
        self.prior_scale = prior_scale  # b in Laplace(0, b)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)

        # KL(N(μ,σ²) || Laplace(0,b))
        # = -0.5 - 0.5*log(2πσ²) + log(2b) + (σ² + μ²)^0.5 / b  (approx)
        # More precisely, using numerical expectation of log-ratio:
        b = self.prior_scale
        # E_q[log q] = -0.5*(1 + logvar + log(2π))
        # E_q[log p] = -log(2b) - E[|z|]/b
        # E[|z|] for N(μ,σ²) = σ·√(2/π)·exp(-μ²/2σ²) + μ·erf(μ/σ√2)
        var = logvar.exp()
        abs_mean = std * math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * mu.pow(2) / var) \
                   + mu * torch.erf(mu / (std * math.sqrt(2.0)))
        kl = -0.5 * (1 + logvar) + math.log(2 * b) + abs_mean / b
        kl = kl.sum(-1).mean()

        sparsity = (z.abs() < 0.05).float().mean().item()

        return self.decoder(z), {"z": z, "kl": kl, "sparsity": sparsity}


# ===========================================================================
#  Training
# ===========================================================================

def train_model(
    model, data, name, epochs=2000, lr=3e-4, batch_size=64,
    beta=1.0, device="cuda",
) -> dict:
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

            # Compute regularization per model type
            if name.startswith("beta_vae"):
                reg = beta * info["kl"]
            elif name == "vqvae":
                reg = info["vq_loss"]
            elif name == "spike_slab":
                reg = beta * (info["kl_gauss"] + info["kl_bernoulli"])
            elif name == "sc_vae":
                reg = beta * info["kl"]
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
        best_recon = min(best_recon, avg_recon)

        if epoch % 200 == 0 or epoch == 1:
            extra = ""
            if name == "vqvae":
                extra = f" perp={info['perplexity'].item():.1f}"
            elif "sparsity" in info:
                extra = f" sparse={info['sparsity']:.2%}"
            log.info(
                f"  [{name:>12s}] Epoch {epoch:4d} | "
                f"recon {avg_recon:.4f} | reg {epoch_reg/n_batches:.4f} | "
                f"best {best_recon:.4f}{extra}"
            )

    # Final eval
    model.eval()
    with torch.no_grad():
        all_x = data.tensors[0].to(device)
        x_recon, info = model(all_x)
        final_mse = F.mse_loss(x_recon, all_x, reduction='sum').item() / all_x.size(0)
        z = info.get("z", None)
        sparsity = 0.0
        if z is not None:
            sparsity = (z.abs() < 0.05).float().mean().item()
        if "sparsity" in info:
            sparsity = info["sparsity"]

    return {"recon_mse": final_mse, "best_recon": best_recon, "sparsity": sparsity}


# ===========================================================================
#  Main
# ===========================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    data = make_toy_data()
    X = data.tensors[0]
    log.info(f"Data: {X.shape}")

    results = {}

    # --- β-VAE sweep ---
    for beta_val in [4, 10, 20]:
        name = f"beta_vae_b{beta_val}"
        log.info(f"\n=== β-VAE (β={beta_val}) ===")
        model = BetaVAE(latent_dim=128)
        results[name] = train_model(model, data, "beta_vae", epochs=2000, beta=beta_val, device=device)

    # --- VQ-VAE (small codebook) ---
    for n_emb in [32, 128]:
        name = f"vqvae_{n_emb}"
        log.info(f"\n=== VQ-VAE ({n_emb} embeddings) ===")
        model = VQVAE(n_embeddings=n_emb, embedding_dim=128)
        results[name] = train_model(model, data, "vqvae", epochs=2000, device=device)

    # --- Spike-and-Slab VAE ---
    for beta_val in [0.005, 0.05, 0.5]:
        name = f"spike_slab_b{beta_val}"
        log.info(f"\n=== Spike-and-Slab VAE (β={beta_val}) ===")
        model = SpikeSlabVAE(latent_dim=128)
        results[name] = train_model(model, data, "spike_slab", epochs=2000, beta=beta_val, device=device)

    # --- SC-VAE (Laplace) ---
    for beta_val in [0.005, 0.05, 0.5]:
        name = f"sc_vae_b{beta_val}"
        log.info(f"\n=== SC-VAE Laplace (β={beta_val}) ===")
        model = SCVAE(latent_dim=128, prior_scale=1.0)
        results[name] = train_model(model, data, "sc_vae", epochs=2000, beta=beta_val, device=device)

    # --- Summary ---
    log.info("\n" + "=" * 80)
    log.info("  STRUCTURED BASELINES — COMPARISON RESULTS")
    log.info("=" * 80)
    log.info(f"{'Model':<22s} {'Recon MSE':>12s} {'Best MSE':>12s} {'Sparsity':>10s}")
    log.info("-" * 58)
    for name, r in sorted(results.items(), key=lambda x: x[1]['best_recon']):
        log.info(
            f"{name:<22s} {r['recon_mse']:>12.4f} {r['best_recon']:>12.4f} "
            f"{r['sparsity']:>9.2%}"
        )
    log.info("-" * 58)
    log.info("Compare with Hybrid Sparse VAE: best_recon=1.82, sparsity=68%")
    log.info("")
    log.info("EXPECTED RANKING (structured models):")
    log.info("  VQ-VAE(128) < SC-VAE(β=0.005) < Hybrid_VAE ≈ Spike-Slab < β-VAE(β=20)")


if __name__ == "__main__":
    main()
