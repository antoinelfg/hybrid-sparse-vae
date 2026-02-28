"""Dictionary matrix **A** for the sparse coding layer.

The dictionary maps atom activations to the physical latent space:
:math:`z = A \\cdot B` where :math:`B = \\gamma \\odot \\delta`.

Supports optional column-norm constraints (so energy is controlled
exclusively by the magnitudes :math:`\\gamma`) and DCT-based
initialization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DictionaryMatrix(nn.Module):
    """Learnable dictionary matrix **A** ∈ ℝ^{latent_dim × n_atoms}.

    Parameters
    ----------
    n_atoms : int
        Number of dictionary atoms (columns of A).
    latent_dim : int
        Physical latent dimension (rows of A, e.g. frequency bins).
    normalize : bool
        If ``True``, columns are L2-normalised *after* each forward
        pass so that the signal energy is carried entirely by
        the magnitudes γ.
    init : str
        Initialization strategy: ``"dct"`` (Discrete Cosine Transform
        atoms), ``"random"`` (Xavier-uniform), or ``"identity"``
        (zero-padded identity, only when n_atoms ≤ latent_dim).
    """

    def __init__(
        self,
        n_atoms: int,
        latent_dim: int,
        normalize: bool = True,
        init: str = "dct",
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.normalize = normalize

        # Raw weight matrix
        weight = torch.empty(latent_dim, n_atoms)
        if init == "dct":
            weight = self._dct_init(latent_dim, n_atoms)
        elif init == "identity":
            weight = self._identity_init(latent_dim, n_atoms)
        else:  # "random"
            nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

    # ------------------------------------------------------------------
    #  Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dct_init(rows: int, cols: int) -> Tensor:
        """Type-II DCT basis vectors as initial atoms."""
        n = torch.arange(rows, dtype=torch.float32)
        k = torch.arange(cols, dtype=torch.float32)
        # DCT-II: cos(π(2n+1)k / 2N)
        basis = torch.cos(math.pi * (2 * n.unsqueeze(1) + 1) * k.unsqueeze(0) / (2 * rows))
        # Normalise columns
        basis = F.normalize(basis, p=2, dim=0)
        return basis

    @staticmethod
    def _identity_init(rows: int, cols: int) -> Tensor:
        """Padded identity (works best when n_atoms ≤ latent_dim)."""
        weight = torch.zeros(rows, cols)
        m = min(rows, cols)
        weight[:m, :m] = torch.eye(m)
        return weight

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, B: Tensor) -> Tensor:
        """Project atom activations to physical latent space.

        Parameters
        ----------
        B : Tensor — ``[..., n_atoms]``
            Sparse signed activations :math:`\\gamma \\odot \\delta`.

        Returns
        -------
        z : Tensor — ``[..., latent_dim]``
        """
        W = self.weight
        if self.normalize:
            W = F.normalize(W, p=2, dim=0)
        # F.linear(B, W) computes B @ W^T.  W is [latent_dim, n_atoms],
        # so W^T is [n_atoms, latent_dim] → z = [B, latent_dim].  ✓
        z = F.linear(B, W)
        return z

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def get_atoms(self) -> Tensor:
        """Return the (optionally normalised) atom matrix [latent_dim, n_atoms]."""
        W = self.weight.data
        if self.normalize:
            W = F.normalize(W, p=2, dim=0)
        return W.detach()
