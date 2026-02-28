"""Core mathematical operators for the Hybrid Sparse VAE.

This package contains differentiable sampling routines and numerical
utilities that implement the theoretical foundations (IRG for Gamma k<1,
hypergeometric series).
"""

from .implicit_gamma import sample_implicit_gamma, ImplicitGamma
from .series_utils import log_hypergeometric_series

__all__ = [
    "sample_implicit_gamma",
    "ImplicitGamma",
    "log_hypergeometric_series",
]
