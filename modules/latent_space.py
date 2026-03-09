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


def sample_gumbel_truncated(logits: Tensor, tau: float, mode: str = "ternary", epsilon: float = 0.05) -> Tensor:
    """
    Truncated Straight-Through Gumbel-Softmax for both binary and ternary modes.
    logits: [..., 2] for binary, [..., 3] for ternary.
    """
    # 1. Compute raw probabilities
    probs = F.softmax(logits, dim=-1)
    
    # 2. Standard Gumbel-Softmax
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbel_softmax = F.softmax((logits + gumbels) / tau, dim=-1)
    
    # 3. Straight-Through Argmax
    indices = gumbel_softmax.max(dim=-1, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
    y_st = (y_hard - gumbel_softmax).detach() + gumbel_softmax
    
    # Map to actual values depending on the mode
    if mode == "ternary":
        values = torch.tensor([-1.0, 0.0, 1.0], device=logits.device)
        zero_index = 1
    elif mode == "binary":
        values = torch.tensor([0.0, 1.0], device=logits.device)
        zero_index = 0
    else:
        raise ValueError(f"Unknown structure mode: {mode}")
        
    delta = torch.sum(y_st * values, dim=-1)
    
    # 4. Truncation Mask (The Anti-Pollution Shield)
    zero_prob = probs[..., zero_index] 
    mask = (zero_prob < (1.0 - epsilon)).float()
    
    return delta * mask


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax projection on the probability simplex along ``dim``."""
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cssv = z_sorted.cumsum(dim) - 1
    rho = torch.arange(1, z.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    view_shape = [1] * z.dim()
    view_shape[dim] = -1
    rho = rho.view(view_shape)
    support = z_sorted > (cssv / rho)
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = cssv.gather(dim, k - 1) / k.to(logits.dtype)
    return torch.clamp(z - tau, min=0.0)


def entmax_bisect(logits: Tensor, *, alpha: float = 1.5, dim: int = -1, n_iter: int = 50) -> Tensor:
    """Alpha-entmax via bisection, suitable for small categorical heads."""
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"entmax alpha must be in (1, 2], got {alpha}")
    z = logits - logits.max(dim=dim, keepdim=True).values
    scale = alpha - 1.0
    power = 1.0 / scale
    tau_lo = z.max(dim=dim, keepdim=True).values - 1.0 / scale
    tau_hi = z.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) * 0.5
        probs = torch.clamp(scale * (z - tau_mid), min=0.0).pow(power)
        sum_probs = probs.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(sum_probs > 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(sum_probs > 1.0, tau_hi, tau_mid)
    tau = (tau_lo + tau_hi) * 0.5
    probs = torch.clamp(scale * (z - tau), min=0.0).pow(power)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def binary_presence_projection(
    presence_logits: Tensor,
    *,
    estimator: str,
    sampling: str,
    temp: float,
    tau_presence_eval: float,
    presence_alpha: float,
) -> tuple[Tensor, Tensor]:
    """Project absent/present logits independently for each atom."""
    if estimator == "gumbel_binary":
        if sampling == "soft":
            probs = F.softmax(presence_logits / max(temp, 1e-6), dim=-1)
            presence_probs = probs[..., 1]
            presence_det = (presence_probs > tau_presence_eval).float()
            return presence_probs, presence_det
        if sampling == "deterministic":
            probs = F.softmax(presence_logits, dim=-1)
            presence_det = (presence_logits.argmax(dim=-1) == 1).float()
            return presence_det, presence_det
        gumbels = -torch.empty_like(presence_logits).exponential_().log()
        y_soft = F.softmax((presence_logits + gumbels) / max(temp, 1e-6), dim=-1)
        idx = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(presence_logits).scatter_(-1, idx, 1.0)
        y_st = (y_hard - y_soft).detach() + y_soft
        presence_probs = y_st[..., 1]
        presence_det = (idx.squeeze(-1) == 1).float()
        return presence_probs, presence_det

    scaled_logits = presence_logits / max(temp, 1e-6) if sampling == "soft" else presence_logits
    if estimator == "sparsemax_binary":
        probs = sparsemax(scaled_logits, dim=-1)
    elif estimator == "entmax15_binary":
        probs = entmax_bisect(scaled_logits, alpha=presence_alpha, dim=-1)
    else:
        raise ValueError(f"Unknown presence_estimator: {estimator}")
    presence_probs = probs[..., 1]
    presence_det = (presence_probs > tau_presence_eval).float()
    if sampling == "deterministic":
        return presence_det, presence_det
    return presence_probs, presence_det


def binary_sign_projection(
    sign_logits: Tensor,
    *,
    estimator: str,
    sampling: str,
    temp: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project {-1,+1} logits independently for each atom."""
    if estimator not in {"gumbel_binary", "gumbel_sign_binary"}:
        raise ValueError(f"Unsupported sign_estimator: {estimator}")

    values = torch.tensor([-1.0, 1.0], device=sign_logits.device, dtype=sign_logits.dtype)
    if sampling == "soft":
        probs = F.softmax(sign_logits / max(temp, 1e-6), dim=-1)
        sign_values = probs[..., 1] - probs[..., 0]
        sign_det = torch.where(probs[..., 1] >= probs[..., 0], 1.0, -1.0)
        return sign_values, sign_det, probs[..., 1]
    if sampling == "deterministic":
        idx = sign_logits.argmax(dim=-1)
        sign_det = values[idx]
        sign_probs = F.softmax(sign_logits, dim=-1)[..., 1]
        return sign_det, sign_det, sign_probs

    gumbels = -torch.empty_like(sign_logits).exponential_().log()
    y_soft = F.softmax((sign_logits + gumbels) / max(temp, 1e-6), dim=-1)
    idx = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(sign_logits).scatter_(-1, idx, 1.0)
    y_st = (y_hard - y_soft).detach() + y_soft
    sign_values = torch.sum(y_st * values, dim=-1)
    sign_det = values[idx.squeeze(-1)]
    return sign_values, sign_det, y_soft[..., 1]


def combine_presence_sign(
    presence: Tensor,
    sign_scores: Tensor,
    *,
    sampling: str,
    tau_presence_eval: float,
) -> tuple[Tensor, Tensor]:
    """Combine multi-label presence with signed activations."""
    sign_values = torch.tanh(sign_scores)
    if sampling == "deterministic":
        sign_det = torch.sign(sign_scores)
        return (presence > tau_presence_eval).float() * sign_det, sign_det
    return presence * sign_values, sign_values


def combine_presence_and_sign(
    presence: Tensor,
    sign_values: Tensor,
) -> Tensor:
    """Combine binary presence with binary/soft sign values."""
    return presence * sign_values


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
        k_max: float = float("inf"),      # Optional Gamma shape ceiling
        magnitude_dist: str = "gamma",       # "gamma" or "gaussian"
        structure_mode: str = "ternary",     # "ternary" or "binary"
        temporal_mode: bool = True,          # True: ConvNMF, False: Dense NMF
        delta_factorization: str = "ternary_direct",
        presence_estimator: str = "gumbel_binary",
        sign_estimator: str = "gumbel_binary",
        presence_alpha: float = 1.5,
        tau_presence_eval: float = 0.5,
        gumbel_epsilon: float = 0.05,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        self.k_min = k_min
        self.k_max = k_max

        self.magnitude_dist = magnitude_dist
        self.structure_mode = structure_mode
        self.n_logits = 3 if structure_mode == "ternary" else 2
        self.temporal_mode = temporal_mode
        self.delta_factorization = delta_factorization
        self.presence_estimator = presence_estimator
        self.sign_estimator = sign_estimator
        self.presence_alpha = presence_alpha
        self.tau_presence_eval = tau_presence_eval
        self.gumbel_epsilon = gumbel_epsilon

        # ---- Parameter heads ------------------------------------------
        # Output: k (n_atoms), θ (n_atoms), struct logits (n_atoms × n_logits)
        self.conv_params = nn.Conv1d(input_dim, n_atoms * (2 + self.n_logits), kernel_size=1)

        # ---- Dictionary (only for non-temporal Dense NMF) --------------
        if not self.temporal_mode:
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
        if not self.temporal_mode and h.dim() == 2:
            h = h.unsqueeze(-1)  # [B, C] -> [B, C, 1]
            
        params = self.conv_params(h)  # [B, n_atoms * 5, T]

        # ---------- Split parameters -----------------------------------
        raw_k, raw_theta, logits_struct = torch.split(
            params,
            [self.n_atoms, self.n_atoms, self.n_atoms * self.n_logits],
            dim=1,
        )

        # k and θ must be > 0
        k = F.softplus(raw_k) + self.k_min
        if self.k_max < 1000.0:  # custom ceiling
            k = torch.clamp(k, max=self.k_max)
        else:
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
        # logits_struct shape: [B, n_atoms * n_logits, T]
        B_dim = logits_struct.shape[0]
        T_dim = logits_struct.shape[2]
        
        # Reshape to [B, n_atoms, n_logits, T] and then transpose to [B, n_atoms, T, n_logits]
        # or directly to compute Softmax over the last dimension.
        # It's usually easier to put the n_logits dimension last for the distributions.
        logits = logits_struct.view(B_dim, self.n_atoms, self.n_logits, T_dim)
        logits = logits.permute(0, 1, 3, 2) # [B, n_atoms, T, n_logits]

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
            delta = sample_gumbel_truncated(
                logits=logits, 
                tau=temp, 
                mode=self.structure_mode, 
                epsilon=self.gumbel_epsilon,
            )

        # transpose delta back to [B, n_atoms, T]
        # Currently delta is [B, n_atoms, T] actually, since the last dim was reduced.
        # Pytorch reductions preserve the leading dimensions.
        # Let's verify: probs is [B, n_atoms, T, n_logits]. `probs[..., 2]` is [B, n_atoms, T].
        # So delta is already [B, n_atoms, T]. No transpose needed.

        # ---------- C. Polar factorization -----------------------------
        B = gamma * delta  # sparse signed activations  [B, n_atoms, T]

        # ---------- D. Dictionary projection ------------------
        if self.temporal_mode:
            z = B  # Decoded physically by ConvTranspose1D
        else:
            B = B.squeeze(-1) # [B, n_atoms, 1] -> [B, n_atoms]
            z = self.dictionary(B)  # [B, latent_dim]

        info = {
            "z": z,
            "B": B,
            "gamma": gamma.squeeze(-1) if not self.temporal_mode else gamma,
            "delta": delta.squeeze(-1) if not self.temporal_mode else delta,
            "k": (k if self.magnitude_dist == "gamma" else raw_k).squeeze(-1) if not self.temporal_mode else (k if self.magnitude_dist == "gamma" else raw_k),
            "theta": theta.squeeze(-1) if not self.temporal_mode else theta,
            "logits": logits.squeeze(2) if not self.temporal_mode else logits,
        }
        return z, info

    def forward_from_params(
        self,
        k: Tensor,
        theta: Tensor,
        logits: Tensor,
        temp: float = 1.0,
        sampling: str = "stochastic",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Sampling-only forward pass for pre-computed parameters (e.g. from a LISTA encoder).

        Skips the ``conv_params`` projection head entirely and goes straight
        to the Gamma / Gumbel-Softmax sampling stage.

        Parameters
        ----------
        k : Tensor — ``[B, n_atoms, T']``   Gamma shape (already softplus'd and clamped)
        theta : Tensor — ``[B, n_atoms, T']`` Gamma scale (already softplus'd)
        logits : Tensor — ``[B, n_atoms, T', n_classes]``  Ternary/binary logits
        """
        # ---------- A. Magnitude -----------------------------------------
        if self.magnitude_dist == "gaussian":
            if sampling in ("soft", "deterministic"):
                gamma = k
            else:
                gamma = k + theta * torch.randn_like(k)
        else:
            if sampling in ("soft", "deterministic"):
                gamma = k * theta
            else:
                gamma = sample_implicit_gamma(k, theta)

        # ---------- B. Structure -----------------------------------------
        if sampling == "soft":
            probs = F.softmax(logits / temp, dim=-1)
            if self.structure_mode == "ternary":
                delta = probs[..., 2] - probs[..., 0]
            else:
                delta = probs[..., 1]
        elif sampling == "deterministic":
            idx = logits.argmax(dim=-1)
            delta_one_hot = F.one_hot(idx, self.n_logits).float()
            if self.structure_mode == "ternary":
                delta = delta_one_hot[..., 2] - delta_one_hot[..., 0]
            else:
                delta = delta_one_hot[..., 1]
        else:
            delta = sample_gumbel_truncated(
                logits=logits,
                tau=temp,
                mode=self.structure_mode,
                epsilon=self.gumbel_epsilon,
            )

        # ---------- C. Polar factorization --------------------------------
        B = gamma * delta   # [B, n_atoms, T']

        if self.temporal_mode:
            z = B
        else:
            B = B.squeeze(-1)
            z = self.dictionary(B)

        info = {
            "z": z,
            "B": B,
            "gamma": gamma,
            "delta": delta,
            "k": k,
            "theta": theta,
            "logits": logits,
        }
        return z, info

    def forward_from_factorized_params(
        self,
        k: Tensor,
        theta: Tensor,
        presence_logits: Tensor,
        sign_scores: Tensor,
        temp: float = 1.0,
        sampling: str = "stochastic",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Sampling forward pass for ``presence × sign`` structure factorization."""
        if self.magnitude_dist == "gaussian":
            if sampling in ("soft", "deterministic"):
                gamma = k
            else:
                gamma = k + theta * torch.randn_like(k)
        else:
            if sampling in ("soft", "deterministic"):
                gamma = k * theta
            else:
                gamma = sample_implicit_gamma(k, theta)

        presence_probs, presence_det = binary_presence_projection(
            presence_logits,
            estimator=self.presence_estimator,
            sampling=sampling,
            temp=temp,
            tau_presence_eval=self.tau_presence_eval,
            presence_alpha=self.presence_alpha,
        )
        delta, sign_values = combine_presence_sign(
            presence_probs,
            sign_scores,
            sampling=sampling,
            tau_presence_eval=self.tau_presence_eval,
        )

        B = gamma * delta
        if self.temporal_mode:
            z = B
        else:
            B = B.squeeze(-1)
            z = self.dictionary(B)

        info = {
            "z": z,
            "B": B,
            "gamma": gamma.squeeze(-1) if not self.temporal_mode else gamma,
            "delta": delta.squeeze(-1) if not self.temporal_mode else delta,
            "k": k.squeeze(-1) if not self.temporal_mode else k,
            "theta": theta.squeeze(-1) if not self.temporal_mode else theta,
            "logits": presence_logits.squeeze(2) if not self.temporal_mode else presence_logits,
            "presence_probs": presence_probs.squeeze(-1) if presence_probs.dim() == 3 and not self.temporal_mode else presence_probs,
            "presence_det": presence_det.squeeze(-1) if presence_det.dim() == 3 and not self.temporal_mode else presence_det,
            "sign_values": sign_values.squeeze(-1) if not self.temporal_mode else sign_values,
            "delta_factorization": self.delta_factorization,
            "presence_estimator": self.presence_estimator,
        }
        return z, info

    def forward_from_fully_factorized_params(
        self,
        k: Tensor,
        theta_tilde: Tensor,
        input_scale: Tensor,
        presence_logits: Tensor,
        sign_logits: Tensor,
        temp: float = 1.0,
        sampling: str = "stochastic",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Sampling forward pass for the fully polar ``gamma × presence × sign`` factorization."""
        theta_final = theta_tilde * input_scale

        if self.magnitude_dist == "gaussian":
            if sampling in ("soft", "deterministic"):
                gamma = k
            else:
                gamma = k + theta_final * torch.randn_like(k)
        else:
            if sampling in ("soft", "deterministic"):
                gamma = k * theta_final
            else:
                gamma = sample_implicit_gamma(k, theta_final)

        presence_probs, presence_det = binary_presence_projection(
            presence_logits,
            estimator=self.presence_estimator,
            sampling=sampling,
            temp=temp,
            tau_presence_eval=self.tau_presence_eval,
            presence_alpha=self.presence_alpha,
        )
        sign_values, sign_det, sign_probs = binary_sign_projection(
            sign_logits,
            estimator=self.sign_estimator,
            sampling=sampling,
            temp=temp,
        )
        delta = combine_presence_and_sign(presence_probs, sign_values)

        B = gamma * delta
        if self.temporal_mode:
            z = B
        else:
            B = B.squeeze(-1)
            z = self.dictionary(B)

        info = {
            "z": z,
            "B": B,
            "gamma": gamma.squeeze(-1) if not self.temporal_mode else gamma,
            "delta": delta.squeeze(-1) if not self.temporal_mode else delta,
            "k": k.squeeze(-1) if not self.temporal_mode else k,
            "theta": theta_final.squeeze(-1) if not self.temporal_mode else theta_final,
            "theta_tilde": theta_tilde.squeeze(-1) if not self.temporal_mode else theta_tilde,
            "theta_final": theta_final.squeeze(-1) if not self.temporal_mode else theta_final,
            "presence_logits": presence_logits.squeeze(2) if not self.temporal_mode else presence_logits,
            "sign_logits": sign_logits.squeeze(2) if not self.temporal_mode else sign_logits,
            "logits": presence_logits.squeeze(2) if not self.temporal_mode else presence_logits,
            "presence_probs": presence_probs.squeeze(-1) if presence_probs.dim() == 3 and not self.temporal_mode else presence_probs,
            "presence_det": presence_det.squeeze(-1) if presence_det.dim() == 3 and not self.temporal_mode else presence_det,
            "sign_values": sign_values.squeeze(-1) if sign_values.dim() == 3 and not self.temporal_mode else sign_values,
            "sign_det": sign_det.squeeze(-1) if sign_det.dim() == 3 and not self.temporal_mode else sign_det,
            "sign_probs": sign_probs.squeeze(-1) if sign_probs.dim() == 3 and not self.temporal_mode else sign_probs,
            "input_scale": input_scale.squeeze(-1) if input_scale.dim() > 1 else input_scale,
            "delta_factorization": "presence_sign",
            "presence_estimator": self.presence_estimator,
            "sign_estimator": self.sign_estimator,
        }
        return z, info
