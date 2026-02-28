"""Vectorized hypergeometric series utilities.

Provides stable, log-space evaluation of the confluent hypergeometric
series that appears in the derivative of the incomplete Gamma CDF with
respect to the shape parameter k.

Reference
---------
K. O. Geddes, M. L. Glasser, R. A. Moore, and H. J. Scott,
"Evaluation of classes of definite integrals involving elementary
functions via differentiation of special functions",
*Appl. Algebra Engrg. Comm. Comput.*, 1990.
"""

from __future__ import annotations

import torch
from torch import Tensor


def log_hypergeometric_series(
    k: Tensor,
    x: Tensor,
    n_terms: int = 50,
) -> Tensor:
    r"""Evaluate the log-space hypergeometric series arising in
    :math:`\partial P(k, x)/\partial k`.

    The series is

    .. math::
        S(k, x) = \sum_{n=0}^{N-1} \frac{(-1)^n\, x^{k+n}}{n!\,(k+n)}
                   \!\left[\ln x - \frac{1}{k+n}\right].

    All intermediate products are computed in **log-space** to avoid
    underflow when :math:`x \to 0` and :math:`k < 1`.

    Parameters
    ----------
    k : Tensor
        Shape parameter, broadcastable with *x*.  Expected shape
        ``[..., Atoms]``.
    x : Tensor
        Standardized variable :math:`z/\theta`, same shape requirements
        as *k*.
    n_terms : int, optional
        Number of terms in the truncated series (default 30).

    Returns
    -------
    S : Tensor
        Series value, same leading shape as *k*.
    """
    # Add trailing dim for broadcasting with n  →  [..., Atoms, 1]
    k_ = k.unsqueeze(-1)
    x_ = x.unsqueeze(-1)

    n = torch.arange(n_terms, device=k.device, dtype=k.dtype)  # [N]

    log_x = torch.log(x_ + 1e-12)  # safety epsilon
    log_fac_n = torch.lgamma(n + 1)  # log(n!)
    kpn = k_ + n  # k + n

    # log |term_n| = (k+n) ln(x) - log(n!) - log(k+n)
    log_abs_term = kpn * log_x - log_fac_n - torch.log(kpn)

    sign = (-1.0) ** n  # alternating sign

    brackets = log_x - 1.0 / kpn

    terms = sign * torch.exp(log_abs_term) * brackets  # [..., Atoms, N]

    S = terms.sum(dim=-1)  # sum over n  →  [..., Atoms]
    return S
