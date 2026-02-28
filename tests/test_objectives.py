"""Validation tests for utils/objectives.py.

Compares our closed-form kl_gamma against PyTorch's built-in
torch.distributions.kl_divergence for the Gamma distribution.
This is the definitive check that our formula matches the canonical
implementation.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as dist

from utils.objectives import kl_gamma, kl_categorical, compute_hybrid_loss


class TestKLGamma:
    """Validate kl_gamma against torch.distributions.kl_divergence."""

    @pytest.mark.parametrize(
        "k_val, theta_val, k0_val, theta0_val",
        [
            (0.3, 1.0, 0.2, 1.0),    # sub-1 k, prior match
            (0.5, 2.0, 0.2, 1.0),    # k<1, different scales
            (1.0, 1.0, 1.0, 1.0),    # same → KL should be 0
            (2.0, 0.5, 1.0, 1.0),    # k>1
            (0.1, 3.0, 0.5, 0.5),    # extreme mismatch
            (5.0, 0.1, 0.2, 1.0),    # large k, small θ
        ],
    )
    def test_matches_pytorch(self, k_val, theta_val, k0_val, theta0_val):
        """Our kl_gamma(k, θ, k₀, θ₀) must match PyTorch's KL
        for Gamma(concentration=k, rate=1/θ)."""
        k = torch.tensor([k_val])
        theta = torch.tensor([theta_val])
        k_0 = torch.tensor([k0_val])
        theta_0 = torch.tensor([theta0_val])

        # Our implementation
        our_kl = kl_gamma(k, theta, k_0, theta_0)

        # PyTorch reference (Gamma uses concentration, rate)
        q = dist.Gamma(concentration=k, rate=1.0 / theta)
        p = dist.Gamma(concentration=k_0, rate=1.0 / theta_0)
        ref_kl = dist.kl_divergence(q, p).sum()

        assert torch.allclose(our_kl, ref_kl, atol=1e-5), (
            f"k={k_val}, θ={theta_val}, k₀={k0_val}, θ₀={theta0_val}: "
            f"ours={our_kl.item():.8f} vs ref={ref_kl.item():.8f}"
        )

    def test_kl_same_distribution_is_zero(self):
        k = torch.tensor([1.5])
        theta = torch.tensor([2.0])
        kl = kl_gamma(k, theta, 1.5, 2.0)
        assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_kl_nonnegative(self):
        """KL divergence must always be ≥ 0."""
        torch.manual_seed(123)
        k = torch.rand(50) * 3 + 0.1
        theta = torch.rand(50) * 3 + 0.1
        k_0 = torch.rand(50) * 3 + 0.1
        theta_0 = torch.rand(50) * 3 + 0.1

        # Element-wise KL (before .sum())
        t1 = (k - k_0) * torch.digamma(k)
        t2 = -torch.lgamma(k) + torch.lgamma(k_0)
        t3 = k_0 * (torch.log(theta_0) - torch.log(theta))
        t4 = k * (theta / theta_0 - 1.0)
        kl_elems = t1 + t2 + t3 + t4

        assert (kl_elems >= -1e-6).all(), (
            f"Negative KL elements: {kl_elems[kl_elems < -1e-6]}"
        )


class TestKLCategorical:
    def test_uniform_prior(self):
        """KL with uniform prior [1/3, 1/3, 1/3] should be ≥ 0."""
        logits = torch.randn(4, 8, 3)
        prior = torch.tensor([1/3, 1/3, 1/3])
        kl = kl_categorical(logits, prior)
        assert kl.item() >= 0

    def test_matching_gives_zero(self):
        """If q probabilities match p, KL ≈ 0."""
        prior = torch.tensor([0.05, 0.90, 0.05])
        # logits that give approximately the prior probs
        logits = torch.log(prior).unsqueeze(0).expand(10, -1)
        kl = kl_categorical(logits, prior)
        assert kl.item() < 1e-4, f"KL should be ~0, got {kl.item()}"
