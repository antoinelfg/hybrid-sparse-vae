"""Smoke tests for the StructuredLatentSpace module.

Validates:
1. Output shapes are correct.
2. Forward + backward pass completes without errors or NaN.
3. Dictionary normalization is applied.
"""

from __future__ import annotations

import torch
from modules.latent_space import StructuredLatentSpace


class TestStructuredLatentSpace:

    def setup_method(self):
        self.input_dim = 256
        self.n_atoms = 32
        self.latent_dim = 64
        self.batch = 4

        self.module = StructuredLatentSpace(
            input_dim=self.input_dim,
            n_atoms=self.n_atoms,
            latent_dim=self.latent_dim,
        )

    def test_output_shapes(self):
        h = torch.randn(self.batch, self.input_dim)
        z, info = self.module(h, temp=1.0)

        assert z.shape == (self.batch, self.latent_dim)
        assert info["B"].shape == (self.batch, self.n_atoms)
        assert info["gamma"].shape == (self.batch, self.n_atoms)
        assert info["delta"].shape == (self.batch, self.n_atoms)
        assert info["k"].shape == (self.batch, self.n_atoms)
        assert info["theta"].shape == (self.batch, self.n_atoms)
        assert info["logits"].shape == (self.batch, self.n_atoms, 3)

    def test_no_nan_forward(self):
        h = torch.randn(self.batch, self.input_dim)
        z, info = self.module(h, temp=0.5)

        assert not z.isnan().any(), "NaN in z"
        assert not info["gamma"].isnan().any(), "NaN in gamma"
        assert not info["B"].isnan().any(), "NaN in B"

    def test_backward_runs(self):
        h = torch.randn(self.batch, self.input_dim, requires_grad=True)
        z, info = self.module(h, temp=1.0)
        loss = z.sum()
        loss.backward()

        assert h.grad is not None, "No gradient on input"
        assert not h.grad.isnan().any(), "NaN in input gradient"

    def test_delta_ternary(self):
        """Delta values should be in {-1, 0, +1}."""
        h = torch.randn(self.batch, self.input_dim)
        _, info = self.module(h, temp=1.0)
        delta = info["delta"]
        valid = (delta == -1) | (delta == 0) | (delta == 1)
        assert valid.all(), f"Delta contains non-ternary values: {delta.unique()}"

    def test_dictionary_normalization(self):
        """Atom columns should have unit L2 norm when normalize=True."""
        atoms = self.module.dictionary.get_atoms()  # [latent_dim, n_atoms]
        norms = atoms.norm(p=2, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Column norms: {norms}"
        )
