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
        magnitude_dist: str = "gamma",       # "gamma" or "gaussian"
        structure_mode: str = "ternary",     # "ternary" or "binary"
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.k_min = k_min

        self.magnitude_dist = magnitude_dist
        self.structure_mode = structure_mode
        self.n_logits = 3 if structure_mode == "ternary" else 2

        # ---- Parameter heads ------------------------------------------
        # Output: k (n_atoms), θ (n_atoms), struct logits (n_atoms × n_logits)
        self.fc_params = nn.Linear(input_dim, n_atoms * (2 + self.n_logits))

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
        raw_k, raw_theta, logits_struct = torch.split(
            params,
            [self.n_atoms, self.n_atoms, self.n_atoms * self.n_logits],
            dim=-1,
        )

        # k and θ must be > 0
        k = F.softplus(raw_k) + self.k_min
        k = torch.clamp(k, max=1000.0)
        
        theta = F.softplus(raw_theta) + 1e-6
        theta = torch.clamp(theta, min=1e-4, max=100.0)

        # ---------- A. Magnitude --------------------------------------
        if self.magnitude_dist == "gaussian":
            # For ablation: k=mean, theta=std
            if sampling == "soft" or sampling == "deterministic":
                gamma = raw_k  # use raw for unbounded mean
            else:
                gamma = raw_k + theta * torch.randn_like(raw_k)
        else:
            if sampling == "soft" or sampling == "deterministic":
                gamma = k * theta
            else:  # "stochastic"
                gamma = sample_implicit_gamma(k, theta)

        # ---------- B. Structure --------------------------------------
        leading_shape = logits_struct.shape[:-1]
        logits = logits_struct.view(*leading_shape, self.n_atoms, self.n_logits)

        if sampling == "soft":
            probs = F.softmax(logits / temp, dim=-1)
            if self.structure_mode == "ternary":
                delta = probs[..., 2] - probs[..., 0]
            else: # binary [0, 1]
                delta = probs[..., 1]
        elif sampling == "deterministic":
            idx = logits.argmax(dim=-1)
            delta_one_hot = F.one_hot(idx, self.n_logits).float()
            if self.structure_mode == "ternary":
                delta = delta_one_hot[..., 2] - delta_one_hot[..., 0]
            else:
                delta = delta_one_hot[..., 1]
        else:  # "stochastic"
            delta_one_hot = F.gumbel_softmax(logits, tau=temp, hard=True)
            if self.structure_mode == "ternary":
                delta = delta_one_hot[..., 2] - delta_one_hot[..., 0]
            else:
                delta = delta_one_hot[..., 1]

        # ---------- C. Polar factorization -----------------------------
        B = gamma * delta  # sparse signed activations  [..., n_atoms]

        # ---------- D. Dictionary projection ---------------------------
        z = self.dictionary(B)  # [..., latent_dim]

        info = {
            "z": z,
            "B": B,
            "gamma": gamma,
            "delta": delta,
            "k": k if self.magnitude_dist == "gamma" else raw_k, # For Gaussian, k=mean
            "theta": theta,
            "logits": logits,
        }
        return z, info

