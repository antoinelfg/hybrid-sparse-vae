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
    t3 = k_0 * (torch.log(theta) - torch.log(theta_0))
    t4 = k * (theta_0 / theta - 1.0)

    kl = t1 + t2 + t3 + t4  # element-wise
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
        prior_probs = torch.tensor(
            [0.05, 0.90, 0.05], dtype=logits.dtype, device=logits.device
        )

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

    Returns
    -------
    loss : Tensor
        Scalar total loss.
    components : dict
        ``{"recon", "kl_gamma", "kl_delta"}`` for logging.
    """
    # 1. Reconstruction loss (sum over all dims, mean over batch)
    batch_size = x.size(0)
    recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size

    # 2. KL Gamma
    kl_g = kl_gamma(params["k"], params["theta"], k_0, theta_0) / batch_size

    # 3. KL Categorical
    kl_d = kl_categorical(params["logits"], prior_probs) / batch_size

    total = recon_loss + beta_gamma * kl_g + beta_delta * kl_d

    return total, {
        "recon": recon_loss.detach(),
        "kl_gamma": kl_g.detach(),
        "kl_delta": kl_d.detach(),
    }
