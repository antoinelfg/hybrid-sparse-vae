"""Exact loss functions for the Hybrid Sparse VAE.

All KL-divergence terms are computed in **closed form** (no Monte-Carlo
estimation), which is critical for the theoretical argument in the paper.

Formulae
--------
KL(Gamma(k, θ) ‖ Gamma(k₀, θ₀))  (shape/scale parameterization):

    (k - k₀) ψ(k)  −  ln Γ(k) + ln Γ(k₀)
    + k₀ [ln θ − ln θ₀]  +  k (θ₀/θ − 1)

    **Note**: θ is the *scale* parameter throughout this code
    (mean = k·θ).  PyTorch's Gamma uses concentration / rate, so
    rate = 1/θ.

KL(Categorical(q) ‖ Categorical(p)):
    Standard discrete KL.
"""

from __future__ import annotations

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
#  KL Divergence: Gamma
# ---------------------------------------------------------------------------

def kl_gamma(
    k: Tensor,
    theta: Tensor,
    k_0: Tensor | float,
    theta_0: Tensor | float,
) -> Tensor:
    """Exact KL(Gamma(k, θ) ‖ Gamma(k₀, θ₀)), summed over all elements.

    All tensors are broadcastable.
    """
    k_0 = torch.as_tensor(k_0, dtype=k.dtype, device=k.device)
    theta_0 = torch.as_tensor(theta_0, dtype=k.dtype, device=k.device)

    t1 = (k - k_0) * torch.digamma(k)
    t2 = -torch.lgamma(k) + torch.lgamma(k_0)

    # SCALE parameterization: θ = scale (mean = k·θ)
    # t3: k₀ · (ln θ₀ - ln θ)
    t3 = k_0 * (torch.log(theta_0) - torch.log(theta))
    # t4: k · (θ/θ₀ - 1)
    t4 = k * (theta / theta_0 - 1.0)

    kl = t1 + t2 + t3 + t4  # element-wise
    return kl.sum()


# ---------------------------------------------------------------------------
#  KL Divergence: Gaussian (for ablations)
# ---------------------------------------------------------------------------

def kl_gaussian(
    mu: Tensor,
    std: Tensor,
    prior_mu: Tensor | float = 0.0,
    prior_std: Tensor | float = 1.0,
) -> Tensor:
    """Exact KL(N(mu, std^2) ‖ N(prior_mu, prior_std^2)), summed over all elements."""
    prior_mu = torch.as_tensor(prior_mu, dtype=mu.dtype, device=mu.device)
    prior_std = torch.as_tensor(prior_std, dtype=mu.dtype, device=mu.device)
    
    var = std.pow(2)
    prior_var = prior_std.pow(2)
    
    kl = torch.log(prior_std) - torch.log(std) + (var + (mu - prior_mu).pow(2)) / (2 * prior_var) - 0.5
    return kl.sum()


# ---------------------------------------------------------------------------
#  KL Divergence: Categorical (ternary structure)
# ---------------------------------------------------------------------------

def kl_categorical(
    logits: Tensor,
    prior_probs: Tensor | None = None,
) -> Tensor:
    """KL(q ‖ p) for ternary indicators, summed over all elements.

    Parameters
    ----------
    logits : Tensor — ``[..., 3]``
        Unnormalized log-probabilities of {-1, 0, +1}.
    prior_probs : Tensor | None
        Prior class probabilities ``[3]``.
        Default: sparsity-inducing ``[0.05, 0.90, 0.05]``.
    """
    if prior_probs is None:
        if logits.shape[-1] == 3:
            prior_probs = torch.tensor(
                [0.05, 0.90, 0.05], dtype=logits.dtype, device=logits.device
            )
        elif logits.shape[-1] == 2:
            prior_probs = torch.tensor(
                [0.90, 0.10], dtype=logits.dtype, device=logits.device
            )
        else:
            raise ValueError(f"Unsupported logits dimension: {logits.shape[-1]}")

    q = dist.Categorical(logits=logits)
    p = dist.Categorical(probs=prior_probs)
    return dist.kl_divergence(q, p).sum()


# ---------------------------------------------------------------------------
#  Combined ELBO Loss
# ---------------------------------------------------------------------------

def compute_hybrid_loss(
    x: Tensor,
    x_recon: Tensor,
    params: dict[str, Tensor],
    k_0: float = 1.0,
    theta_0: float = 1.0,
    prior_probs: Tensor | None = None,
    beta_gamma: float = 1.0,
    beta_delta: float = 1.0,
    magnitude_dist: str = "gamma",
    masked_recon: bool = False,
    lambda_silence: float = 0.1,
    lambda_recon_l1: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Full ELBO = Reconstruction + β_γ · KL_Gamma + β_δ · KL_Categorical.

    Parameters
    ----------
    x : Tensor
        Original input.
    x_recon : Tensor
        Decoder output.
    params : dict
        Output of :meth:`StructuredLatentSpace.forward` containing
        ``"k"``, ``"theta"``, ``"logits"``.
    k_0, theta_0 : float
        Prior Gamma shape/scale.
    prior_probs : Tensor | None
        Prior for ternary Categorical ([0.05, 0.90, 0.05] by default).
    beta_gamma, beta_delta : float
        KL weighting coefficients (β-VAE style).
    masked_recon : bool
        If True, use masked reconstruction loss separating signal from silence:
          - Signal loss:  MSE averaged only over non-zero bins (M=1)
          - Silence loss: λ_silence × MSE averaged only over zero bins (M=0)
        This prevents the model from ignoring sparse peaks by averaging over
        a sea of zeros. Recommended when using spectral denoising (>80% zeros).
    lambda_silence : float
        Weight of the silence suppression term (default 0.1).
        Keeps x_recon near zero where x is zero, preventing hallucinations.

    Returns
    -------
    loss : Tensor
        Scalar total loss.
    components : dict
        ``{"recon", "kl_gamma", "kl_delta"}`` for logging.
    """
    # 1. Reconstruction loss
    batch_size = x.size(0)

    if masked_recon:
        # Binary mask: 1 where signal is present, 0 where silence
        M = (x > 0).float()

        err = (x_recon - x).pow(2)

        # Normalise by batch_size (same as KL terms) so the signal loss is
        # commensurable with kl_γ and kl_δ regardless of signal density.
        # Dividing by n_signal (pixel-average) gave O(0.01) signal_loss vs
        # O(1-10) KL, causing posterior collapse (all-δ=0 local minimum).
        signal_loss  = (err * M).sum() / batch_size
        silence_loss = (x_recon.pow(2) * (1.0 - M)).sum() / batch_size

        recon_loss = signal_loss + lambda_silence * silence_loss
    else:
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size

    # Optional L1 noise-floor penalty: pulls ALL output pixels toward 0.
    # Acts like spectral pruning — forces decoder to earn every non-zero pixel.
    # Works in concert with masked_recon: signal_loss pulls peaks up,
    # lambda_recon_l1 pulls everything down, silence_loss penalises hallucinations.
    if lambda_recon_l1 > 0.0:
        recon_loss = recon_loss + lambda_recon_l1 * x_recon.abs().sum() / batch_size

    # 2. KL Magnitude (Gamma or Gaussian)
    if magnitude_dist == "gamma":
        kl_g = kl_gamma(params["k"], params["theta"], k_0, theta_0) / batch_size
    elif magnitude_dist == "gaussian":
        kl_g = kl_gaussian(params["k"], params["theta"], prior_mu=0.0, prior_std=1.0) / batch_size
    else:
        raise ValueError(f"Unknown magnitude_dist: {magnitude_dist}")

    # 3. KL Categorical
    kl_d = kl_categorical(params["logits"], prior_probs) / batch_size

    total = recon_loss + beta_gamma * kl_g + beta_delta * kl_d

    return total, {
        "recon": recon_loss.detach(),
        "kl_gamma": kl_g.detach(),
        "kl_delta": kl_d.detach(),
    }

