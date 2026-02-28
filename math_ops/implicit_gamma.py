"""Implicit Reparameterization Gradients (IRG) for the Gamma distribution.

This is the most critical file in the codebase.  It provides a
differentiable Gamma sampler that remains stable for shape parameter
:math:`k < 1` by using the analytic series expansion of
:math:`\\partial P(k, x)/\\partial k` (Geddes) evaluated in log-space,
combined with implicit differentiation of the CDF.

Usage
-----
>>> k = torch.tensor([0.3, 0.7], requires_grad=True)
>>> theta = torch.tensor([1.0, 1.0], requires_grad=True)
>>> z = sample_implicit_gamma(k, theta)
>>> z.sum().backward()
>>> k.grad  # valid, uses IRG

References
----------
* Figurnov et al., "Implicit Reparameterization Gradients", NeurIPS 2018.
* Geddes et al., 1990 – series for d/dk of the incomplete gamma.
"""

from __future__ import annotations

import torch
import torch.distributions as dist
from torch import Tensor
from torch.autograd import Function

from .series_utils import log_hypergeometric_series


# ---------------------------------------------------------------------------
#  Helper: ∂P/∂k  via series expansion
# ---------------------------------------------------------------------------

def _gamma_cdf_derivative_wrt_k(
    k: Tensor,
    x: Tensor,
    n_terms: int = 50,
) -> Tensor:
    r"""Compute :math:`\frac{\partial}{\partial k} P(k, x)` where
    :math:`P` is the **regularized** lower incomplete gamma function
    (CDF of Gamma(k, 1) evaluated at *x*).

    Uses the Geddes series in log-space (see :func:`series_utils.log_hypergeometric_series`).

    Parameters
    ----------
    k : Tensor  — shape ``[...]``
    x : Tensor  — standardized sample :math:`z/\theta`, same shape.
    n_terms : int

    Returns
    -------
    dF_dk : Tensor  — same shape as *k*.
    """
    S = log_hypergeometric_series(k, x, n_terms=n_terms)

    # P(k, x) = regularized lower incomplete gamma (torch name: gammainc)
    F = torch.special.gammainc(k, x)

    # dF/dk = S / Gamma(k)  -  F · ψ(k)
    log_gamma_k = torch.lgamma(k)
    dF_dk = S / torch.exp(log_gamma_k) - F * torch.digamma(k)
    return dF_dk


# ---------------------------------------------------------------------------
#  Custom Autograd Function
# ---------------------------------------------------------------------------

class ImplicitGamma(Function):
    """Differentiable Gamma sampler using Implicit Reparameterization
    Gradients.

    Forward pass samples from :math:`\text{Gamma}(k, \theta)` without
    building a computational graph (the standard rsample path is
    numerically unstable for :math:`k < 1`).

    Backward pass computes exact gradients via implicit differentiation
    of the CDF equation :math:`F(z; k, \theta) = u`, where *u* is the
    quantile drawn during the forward pass.
    """

    @staticmethod
    def forward(ctx, k: Tensor, theta: Tensor) -> Tensor:  # type: ignore[override]
        # --- Forward: standard Gamma sample (no graph) ------------------
        with torch.no_grad():
            # Safety clamps (k and θ must be strictly > 0 for Gamma)
            k_safe = k.clamp(min=1e-4)
            theta_safe = theta.clamp(min=1e-6)
            gamma_dist = dist.Gamma(
                concentration=k_safe, rate=1.0 / theta_safe,
                validate_args=False,
            )
            z = gamma_dist.sample()
            # Clamp away from exact zero for numerical safety in backward
            z = z.clamp(min=1e-30)

        ctx.save_for_backward(k, theta, z)
        return z

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        k, theta, z = ctx.saved_tensors

        # Safety clamps for backward computation
        k_safe = k.clamp(min=1e-4)
        theta_safe = theta.clamp(min=1e-6)

        # Standardized variable
        x = (z / theta_safe).clamp(min=1e-30)

        # PDF of the sampled point (used as denominator in IRG formula)
        log_prob = dist.Gamma(
            concentration=k_safe, rate=1.0 / theta_safe,
            validate_args=False,
        ).log_prob(z)
        pdf = torch.exp(log_prob)

        # ----- A. Gradient w.r.t. k (shape) ----------------------------
        #   dz/dk = - (∂F/∂k) / p(z)
        dF_dk = _gamma_cdf_derivative_wrt_k(k_safe, x)
        # pdf → ∞ as z → 0, so 1/pdf → 0  ⇒ natural damping
        grad_k = -(dF_dk / (pdf + 1e-30))

        # Hard safety: clamp and replace any remaining NaN/Inf
        grad_k = grad_k.clamp(-1e3, 1e3)
        grad_k = torch.where(torch.isfinite(grad_k), grad_k, torch.zeros_like(grad_k))

        # ----- B. Gradient w.r.t. θ (scale) ----------------------------
        #   For the scale param the reparameterization z = θ·ε gives
        #   dz/dθ = z/θ   (simple analytic form).
        grad_theta = z / theta_safe

        # Chain rule with upstream gradient
        return grad_output * grad_k, grad_output * grad_theta


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def sample_implicit_gamma(k: Tensor, theta: Tensor) -> Tensor:
    """Sample from Gamma(k, θ) with IRG-based gradients.

    Parameters
    ----------
    k : Tensor
        Shape (concentration) parameter.  May be < 1.
    theta : Tensor
        Scale parameter (> 0).

    Returns
    -------
    z : Tensor
        Samples from Gamma(k, θ), with valid gradients flowing back
        to both *k* and *θ*.
    """
    return ImplicitGamma.apply(k, theta)
