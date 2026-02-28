"""Structured Latent Space: polar factorization z = A · (γ ⊙ δ).

Orchestrates:
  1. Parameter prediction (encoder output → k, θ, ternary logits)
  2. Magnitude sampling via IRG Gamma (γ)
  3. Structure sampling via Straight-Through Gumbel-Softmax (δ ∈ {-1, 0, +1})
  4. Dictionary projection  z = A · B  where  B = γ ⊙ δ
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from math_ops.implicit_gamma import sample_implicit_gamma
from .dictionary import DictionaryMatrix


class StructuredLatentSpace(nn.Module):
    """Full latent-space module with polar factorization.

    Parameters
    ----------
    input_dim : int
        Dimension of the encoder feature vector feeding into this module.
    n_atoms : int
        Number of dictionary atoms (sparse code width).
    latent_dim : int
        Physical latent dimension (e.g. number of frequency bins).
    dict_init : str
        Dictionary initialization: ``"dct"``, ``"random"``, ``"identity"``.
    normalize_dict : bool
        Whether to L2-normalize dictionary columns.
    k_min : float
        Minimum allowed shape parameter (avoids extreme instability at
        the very start of training).
    """

    def __init__(
        self,
        input_dim: int,
        n_atoms: int,
        latent_dim: int,
        dict_init: str = "dct",
        normalize_dict: bool = True,
        k_min: float = 0.1,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.k_min = k_min

        # ---- Parameter heads ------------------------------------------
        # Output: k (n_atoms), θ (n_atoms), ternary logits (n_atoms × 3)
        self.fc_params = nn.Linear(input_dim, n_atoms * 5)
        # *5 = 1 (k) + 1 (θ) + 3 (ternary logits per atom)

        # ---- Dictionary -----------------------------------------------
        self.dictionary = DictionaryMatrix(
            n_atoms=n_atoms,
            latent_dim=latent_dim,
            normalize=normalize_dict,
            init=dict_init,
        )

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h: Tensor,
        temp: float = 1.0,
        sampling: str = "stochastic",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        h : Tensor — ``[B, input_dim]`` or ``[B, T, input_dim]``
        temp : float
            Temperature for Gumbel-Softmax / softmax.
        sampling : str
            ``"soft"`` — γ=k·θ, δ=continuous softmax (Phase 1: no noise)
            ``"stochastic"`` — Gamma IRG + Gumbel-ST (Phase 2-3: normal VAE)
            ``"deterministic"`` — mean + argmax (eval only)

        Returns
        -------
        z : Tensor — ``[..., latent_dim]``
            Reconstructed latent vector.
        info : dict
            Diagnostic tensors.
        """
        params = self.fc_params(h)  # [..., n_atoms * 5]

        # ---------- Split parameters -----------------------------------
        raw_k, raw_theta, logits_ternary = torch.split(
            params,
            [self.n_atoms, self.n_atoms, self.n_atoms * 3],
            dim=-1,
        )

        # k and θ must be > 0
        k = F.softplus(raw_k) + self.k_min
        theta = F.softplus(raw_theta) + 1e-6

        # ---------- A. Magnitude --------------------------------------
        if sampling == "soft":
            # Gamma mean — fully differentiable, zero noise
            gamma = k * theta
        elif sampling == "deterministic":
            gamma = k * theta
        else:  # "stochastic"
            gamma = sample_implicit_gamma(k, theta)

        # ---------- B. Structure (Ternary delta) ----------------------
        leading_shape = logits_ternary.shape[:-1]
        logits_3 = logits_ternary.view(*leading_shape, self.n_atoms, 3)

        if sampling == "soft":
            # Continuous softmax relaxation — differentiable, no noise
            probs = F.softmax(logits_3 / temp, dim=-1)
            delta = probs[..., 2] - probs[..., 0]  # continuous in [-1, +1]
        elif sampling == "deterministic":
            idx = logits_3.argmax(dim=-1)
            delta_one_hot = F.one_hot(idx, 3).float()
            delta = delta_one_hot[..., 2] - delta_one_hot[..., 0]
        else:  # "stochastic"
            delta_one_hot = F.gumbel_softmax(logits_3, tau=temp, hard=True)
            delta = delta_one_hot[..., 2] - delta_one_hot[..., 0]


        # ---------- C. Polar factorization -----------------------------
        B = gamma * delta  # sparse signed activations  [..., n_atoms]

        # ---------- D. Dictionary projection ---------------------------
        z = self.dictionary(B)  # [..., latent_dim]

        info = {
            "z": z,
            "B": B,
            "gamma": gamma,
            "delta": delta,
            "k": k,
            "theta": theta,
            "logits": logits_3,
        }
        return z, info

