"""Encoder and Decoder building blocks (1-D convolutional ResNets).

These are generic backbones for the Hybrid Sparse VAE. The encoder
maps raw input (e.g. spectral frames) to the parameter space consumed
by :class:`modules.latent_space.StructuredLatentSpace`, while the
decoder reconstructs the input from the latent code **z**.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
#  Residual Block
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """Post-activation residual block for 1-D sequences.

    ``x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU``

    If the channel count changes, a 1×1 skip projection is added.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )

        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.block(x) + self.skip(x))


# ---------------------------------------------------------------------------
#  Encoder
# ---------------------------------------------------------------------------

class Encoder1D(nn.Module):
    """1-D convolutional encoder.

    Outputs a flat feature vector of size *output_dim* per time-frame,
    suitable for feeding into the latent-space parameter heads.

    Parameters
    ----------
    input_channels : int
        Number of input channels (e.g. 1 for waveform, n_mels for mel).
    hidden_channels : list[int]
        Channel widths for successive ResBlocks.
    output_dim : int
        Flat feature dimension per frame.
    temporal_downsample_factor : int | None
        When set, the product of block strides is forced to match this factor.
        This is critical for ConvNMF models where the encoder time reduction
        must stay compatible with the decoder stride.
    """

    @staticmethod
    def _build_stride_schedule(
        target_downsample: int | None,
        n_blocks: int,
    ) -> list[int]:
        if target_downsample is None:
            return [2] * n_blocks
        if target_downsample < 1:
            raise ValueError(f"temporal_downsample_factor must be >= 1, got {target_downsample}")

        remaining = int(target_downsample)
        strides = [1] * n_blocks
        i = 0
        while remaining > 1 and i < n_blocks:
            if remaining % 2 != 0:
                raise ValueError(
                    "Encoder1D only supports power-of-two temporal_downsample_factor "
                    f"with the current ResBlock setup, got {target_downsample}"
                )
            strides[i] = 2
            remaining //= 2
            i += 1

        if remaining != 1:
            raise ValueError(
                "Requested temporal_downsample_factor exceeds encoder capacity: "
                f"target={target_downsample}, max={2 ** n_blocks}"
            )
        return strides

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: list[int] | None = None,
        output_dim: int = 256,
        spatial_pooling: bool = False,
        temporal_downsample_factor: int | None = None,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        layers: list[nn.Module] = []
        stride_schedule = self._build_stride_schedule(
            target_downsample=temporal_downsample_factor,
            n_blocks=len(hidden_channels),
        )
        in_ch = input_channels
        for out_ch, stride in zip(hidden_channels, stride_schedule):
            layers.append(ResBlock1D(in_ch, out_ch, stride=stride))
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)
        self.spatial_pooling = spatial_pooling
        self.stride_schedule = stride_schedule
        self.temporal_downsample_factor = int(torch.tensor(stride_schedule).prod().item())
        if self.spatial_pooling:
            self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv_out = nn.Conv1d(in_ch, output_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  — ``[B, C, T]``

        Returns
        -------
        h : Tensor  — ``[B, output_dim, T']`` or ``[B, output_dim]`` if spatial_pooling
        """
        h = self.backbone(x)        # [B, last_ch, T']
        if self.spatial_pooling:
            h = self.pool(h)        # [B, last_ch, 1]
        h = self.conv_out(h)        # [B, output_dim, T'] or [B, output_dim, 1]
        
        if self.spatial_pooling:
            h = h.squeeze(-1)       # [B, output_dim]
            
        return h


# ---------------------------------------------------------------------------
#  Linear Unrolled LISTA Inference Encoder
# ---------------------------------------------------------------------------

class LinearUnrolledISTAEncoder(nn.Module):
    """Linear LISTA encoder for non-temporal sparse inference.

    This is the dense counterpart of :class:`ConvUnrolledISTAEncoder` for
    toy 1-D signals and other non-temporal regimes. It iteratively refines a
    sparse code with learned recurrence, then projects the final iterate to
    ``k``, ``theta``, and structural logits.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 128,
        n_atoms: int = 64,
        n_iterations: int = 5,
        structure_mode: str = "ternary",
        k_max: float = float("inf"),
        threshold_init: float = 0.1,
        delta_head_mode: str = "shared",
    ):
        super().__init__()
        input_dim = input_channels * input_length
        self.n_iterations = n_iterations
        self.n_atoms = n_atoms
        self.n_classes = 3 if structure_mode == "ternary" else 2
        self.k_max = k_max
        self.delta_head_mode = delta_head_mode

        self.input_proj = nn.Linear(input_dim, n_atoms)
        self.recurrent = nn.Linear(n_atoms, n_atoms, bias=False)
        self.head_k = nn.Linear(n_atoms, n_atoms)
        self.head_theta = nn.Linear(n_atoms, n_atoms)
        self.head_pi = nn.Linear(n_atoms, n_atoms * self.n_classes)
        self.log_threshold = nn.Parameter(torch.full((n_atoms,), math.log(threshold_init)))
        if delta_head_mode == "shared":
            self.structure_proj = None
        elif delta_head_mode == "l2norm":
            self.structure_proj = nn.Linear(n_atoms, n_atoms)
        else:
            raise ValueError(f"Unknown delta_head_mode: {delta_head_mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        drive = self.input_proj(x_flat)
        threshold = self.log_threshold.exp().unsqueeze(0)

        h = torch.sign(drive) * F.relu(drive.abs() - threshold)
        for _ in range(self.n_iterations - 1):
            proposal = drive + self.recurrent(h)
            h = torch.sign(proposal) * F.relu(proposal.abs() - threshold)

        k_out = F.softplus(self.head_k(h)).unsqueeze(-1) + 1e-4
        if self.k_max < float("inf"):
            k_out = torch.clamp(k_out, max=self.k_max)
        theta_out = F.softplus(self.head_theta(h)).unsqueeze(-1) + 1e-4
        if self.delta_head_mode == "l2norm":
            h_struct = F.normalize(h, p=2, dim=1, eps=1e-6)
            h_struct = torch.tanh(self.structure_proj(h_struct))
        else:
            h_struct = h
        logits = self.head_pi(h_struct).view(x.size(0), self.n_atoms, self.n_classes).unsqueeze(2)
        return k_out, theta_out, logits


class PolarLinearLISTAEncoder(nn.Module):
    """Two-branch LISTA encoder with scale-invariant structure and scale-aware energy.

    The structure branch only sees an L2-normalized input, while the energy
    branch receives the raw signal scale plus a detached summary of the
    structure iterate. This mirrors the ``gamma * delta`` factorization in the
    latent space.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 128,
        n_atoms: int = 64,
        n_iterations: int = 5,
        k_min: float = 0.01,
        k_max: float = float("inf"),
        shape_norm: str = "l2_global",
        gain_feature: str = "log_l2",
        shape_detach_to_gamma: bool = True,
        presence_head_bias_init: float = 0.0,
        sign_head_bias_init: float = 0.0,
        threshold_init: float = 0.1,
    ):
        super().__init__()
        if shape_norm != "l2_global":
            raise ValueError(f"Unsupported shape_norm: {shape_norm}")
        if gain_feature not in {"log_l2", "l2"}:
            raise ValueError(f"Unsupported gain_feature: {gain_feature}")

        input_dim = input_channels * input_length
        self.n_iterations = n_iterations
        self.n_atoms = n_atoms
        self.k_min = k_min
        self.k_max = k_max
        self.shape_norm = shape_norm
        self.gain_feature = gain_feature
        self.shape_detach_to_gamma = shape_detach_to_gamma

        self.shape_input_proj = nn.Linear(input_dim, n_atoms)
        self.shape_recurrent = nn.Linear(n_atoms, n_atoms, bias=False)
        self.log_threshold = nn.Parameter(torch.full((n_atoms,), math.log(threshold_init)))

        self.head_presence = nn.Linear(n_atoms, n_atoms * 2)
        self.head_sign = nn.Linear(n_atoms, n_atoms)
        nn.init.constant_(self.head_presence.bias, presence_head_bias_init)
        nn.init.constant_(self.head_sign.bias, sign_head_bias_init)

        gamma_in_dim = input_dim + 1 + n_atoms
        self.gamma_fc1 = nn.Linear(gamma_in_dim, n_atoms)
        self.gamma_fc2 = nn.Linear(n_atoms, n_atoms)
        self.head_k = nn.Linear(n_atoms, n_atoms)
        self.head_theta = nn.Linear(n_atoms, n_atoms)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x_flat = x.view(x.size(0), -1)
        input_scale = x_flat.norm(dim=1, keepdim=True)
        x_shape = x_flat / (input_scale + 1e-6)

        drive = self.shape_input_proj(x_shape)
        threshold = self.log_threshold.exp().unsqueeze(0)
        h_delta = torch.sign(drive) * F.relu(drive.abs() - threshold)
        for _ in range(max(self.n_iterations - 1, 0)):
            proposal = drive + self.shape_recurrent(h_delta)
            h_delta = torch.sign(proposal) * F.relu(proposal.abs() - threshold)

        shape_ctx = h_delta.detach() if self.shape_detach_to_gamma else h_delta
        if self.gain_feature == "log_l2":
            gain_scalar = torch.log(input_scale + 1e-6)
        else:
            gain_scalar = input_scale
        gamma_in = torch.cat([x_flat, gain_scalar, shape_ctx], dim=1)
        h_gamma = F.relu(self.gamma_fc1(gamma_in))
        h_gamma = F.relu(self.gamma_fc2(h_gamma))

        k_out = F.softplus(self.head_k(h_gamma)) + self.k_min
        if self.k_max < float("inf"):
            k_out = torch.clamp(k_out, max=self.k_max)
        theta_out = F.softplus(self.head_theta(h_gamma)) + 1e-4

        presence_logits = self.head_presence(h_delta).view(x.size(0), self.n_atoms, 2).unsqueeze(2)
        sign_scores = self.head_sign(h_delta).unsqueeze(-1)

        return (
            k_out.unsqueeze(-1),
            theta_out.unsqueeze(-1),
            presence_logits,
            sign_scores,
            {
                "input_scale": input_scale.squeeze(-1),
                "shape_state": h_delta,
                "gain_state": h_gamma,
            },
        )


class FullyPolarLinearLISTAEncoder(nn.Module):
    """Strictly polar LISTA encoder with scale-invariant structure and explicit scale injection.

    The structure branch only sees the normalized input ``x / ||x||``. The
    energy branch only sees a detached structure representation and predicts a
    canonical scale ``theta_tilde`` that is later multiplied by the input norm.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 128,
        n_atoms: int = 64,
        n_iterations: int = 5,
        k_min: float = 0.01,
        k_max: float = float("inf"),
        shape_detach_to_gamma: bool = True,
        presence_head_bias_init: float = 0.0,
        sign_head_bias_init: float = 0.0,
        threshold_init: float = 0.1,
    ):
        super().__init__()
        input_dim = input_channels * input_length
        self.n_iterations = n_iterations
        self.n_atoms = n_atoms
        self.k_min = k_min
        self.k_max = k_max
        self.shape_detach_to_gamma = shape_detach_to_gamma

        self.shape_input_proj = nn.Linear(input_dim, n_atoms)
        self.shape_recurrent = nn.Linear(n_atoms, n_atoms, bias=False)
        self.log_threshold = nn.Parameter(torch.full((n_atoms,), math.log(threshold_init)))

        self.head_presence = nn.Linear(n_atoms, n_atoms * 2)
        self.head_sign = nn.Linear(n_atoms, n_atoms * 2)
        nn.init.constant_(self.head_presence.bias, presence_head_bias_init)
        nn.init.constant_(self.head_sign.bias, sign_head_bias_init)

        self.gamma_fc1 = nn.Linear(n_atoms, n_atoms)
        self.gamma_fc2 = nn.Linear(n_atoms, n_atoms)
        self.head_k = nn.Linear(n_atoms, n_atoms)
        self.head_theta = nn.Linear(n_atoms, n_atoms)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x_flat = x.view(x.size(0), -1)
        input_scale = x_flat.norm(dim=1, keepdim=True) + 1e-6
        x_shape = x_flat / input_scale

        drive = self.shape_input_proj(x_shape)
        threshold = self.log_threshold.exp().unsqueeze(0)
        h_delta = torch.sign(drive) * F.relu(drive.abs() - threshold)
        for _ in range(max(self.n_iterations - 1, 0)):
            proposal = drive + self.shape_recurrent(h_delta)
            h_delta = torch.sign(proposal) * F.relu(proposal.abs() - threshold)

        shape_ctx = h_delta.detach() if self.shape_detach_to_gamma else h_delta
        h_gamma = F.relu(self.gamma_fc1(shape_ctx))
        h_gamma = F.relu(self.gamma_fc2(h_gamma))

        k_out = F.softplus(self.head_k(h_gamma)) + self.k_min
        if self.k_max < float("inf"):
            k_out = torch.clamp(k_out, max=self.k_max)
        theta_tilde = F.softplus(self.head_theta(h_gamma)) + 1e-4

        presence_logits = self.head_presence(h_delta).view(x.size(0), self.n_atoms, 2).unsqueeze(2)
        sign_logits = self.head_sign(h_delta).view(x.size(0), self.n_atoms, 2).unsqueeze(2)

        return (
            k_out.unsqueeze(-1),
            theta_tilde.unsqueeze(-1),
            input_scale.unsqueeze(-1),
            presence_logits,
            sign_logits,
            {
                "input_scale": input_scale.squeeze(-1),
                "shape_state": h_delta,
                "gain_state": h_gamma,
                "shape_input": x_shape,
                "theta_canonical": theta_tilde,
            },
        )


# ---------------------------------------------------------------------------
#  Convolutional Unrolled LISTA Inference Encoder
# ---------------------------------------------------------------------------

class ConvUnrolledISTAEncoder(nn.Module):
    """Convolutional LISTA (Learned ISTA) encoder for iterative amortized inference.

    Instead of a single feed-forward pass, the encoder performs ``n_iterations``
    recurrent steps of the form::

        h_t = ReLU(W_x * X + W_h * h_{t-1})

    where:
      - ``W_x`` performs the initial projection (equivalent to Aᵀ X in ISTA)
      - ``W_h`` models mutual inhibition between atoms (equivalent to Aᵀ A)

    After ``n_iterations``, dedicated heads project ``h`` to the parameters
    of the Gamma and Gumbel-Softmax distributions, preserving the time axis.

    This architecture closes the *amortization gap* at the cost of ``n_iterations``
    forward passes through ``W_h``, and is compatible with the ConvNMF decoder
    (``decoder_type="convnmf"``).

    Parameters
    ----------
    input_channels : int
        Number of input channels (e.g., 1 for waveform).
    n_atoms : int
        Dimension of the sparse code / number of atoms in the dictionary.
    n_iterations : int
        Number of LISTA unrolling steps (K in the paper). Default: 3.
    kernel_size : int
        Kernel size for ``W_x`` projection. Should match encoder stride (default 16).
    structure_mode : str
        ``"ternary"`` (3 classes: -1, 0, +1) or ``"binary"`` (2 classes: 0, 1).
    """

    def __init__(
        self,
        input_channels: int = 1,
        n_atoms: int = 64,
        n_iterations: int = 3,
        kernel_size: int = 16,
        structure_mode: str = "ternary",
        k_max: float = float("inf"),   # Clamp Gamma shape to keep sparse regime
        delta_head_mode: str = "shared",
    ):
        super().__init__()
        self.n_iterations = n_iterations
        self.n_atoms = n_atoms
        self.n_classes = 3 if structure_mode == "ternary" else 2
        self.k_max = k_max
        self.delta_head_mode = delta_head_mode

        # W_x: initial projection (preserves time with stride=kernel_size)
        self.W_x = nn.Conv1d(
            input_channels, n_atoms, kernel_size=kernel_size, stride=kernel_size, padding=0, bias=True
        )

        # W_h: lateral inhibition / mutual inhibition between atoms
        self.W_h = nn.Conv1d(n_atoms, n_atoms, kernel_size=1, bias=False)

        # Parameter heads: all operate on the time axis with kernel_size=1
        self.head_k = nn.Conv1d(n_atoms, n_atoms, kernel_size=1)      # Gamma shape (k)
        self.head_theta = nn.Conv1d(n_atoms, n_atoms, kernel_size=1)  # Gamma scale (θ)
        self.head_pi = nn.Conv1d(n_atoms, n_atoms * self.n_classes, kernel_size=1)  # Gumbel logits
        if delta_head_mode == "shared":
            self.structure_proj = None
        elif delta_head_mode == "l2norm":
            self.structure_proj = nn.Conv1d(n_atoms, n_atoms, kernel_size=1)
        else:
            raise ValueError(f"Unknown delta_head_mode: {delta_head_mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor — ``[B, input_channels, T]``

        Returns
        -------
        k_out : Tensor — ``[B, n_atoms, T']``  Gamma shape parameter
        theta_out : Tensor — ``[B, n_atoms, T']``  Gamma scale parameter
        logits : Tensor — ``[B, n_atoms, T', n_classes]``  Gumbel-Softmax logits
        """
        B = x.size(0)

        # 1. Project input into atom space
        h_x = self.W_x(x)    # [B, n_atoms, T']

        # 2. LISTA iterations
        h = torch.zeros_like(h_x)
        for _ in range(self.n_iterations):
            h = F.relu(h_x + self.W_h(h))   # [B, n_atoms, T']

        # 3. Project to distribution parameters
        k_out = F.softplus(self.head_k(h)) + 1e-4        # [B, n_atoms, T']
        if self.k_max < float("inf"):
            k_out = torch.clamp(k_out, max=self.k_max)   # Force sparse Gamma regime
        theta_out = F.softplus(self.head_theta(h)) + 1e-4  # [B, n_atoms, T']

        # 4. Logits: [B, n_atoms * n_classes, T'] → [B, n_atoms, T', n_classes]
        T_prime = h.size(-1)
        if self.delta_head_mode == "l2norm":
            h_struct = F.normalize(h, p=2, dim=1, eps=1e-6)
            h_struct = torch.tanh(self.structure_proj(h_struct))
        else:
            h_struct = h
        logits = self.head_pi(h_struct).view(B, self.n_atoms, self.n_classes, T_prime)
        logits = logits.permute(0, 1, 3, 2).contiguous()  # [B, n_atoms, T', n_classes]

        return k_out, theta_out, logits


# ---------------------------------------------------------------------------
#  Decoder
# ---------------------------------------------------------------------------

class Decoder1D(nn.Module):
    """1-D convolutional decoder (mirror of :class:`Encoder1D`).

    Takes the latent code **z** (or the dictionary projection) and
    produces a reconstruction of the original signal.

    Parameters
    ----------
    latent_dim : int
        Dimension of the input latent vector.
    hidden_channels : list[int]
        Channel widths (in *reverse* order compared to the encoder).
    output_channels : int
        Number of output channels.
    output_length : int
        Target temporal length of the output.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_channels: list[int] | None = None,
        output_channels: int = 1,
        output_length: int = 128,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [256, 128, 64, 32]

        self.init_length = output_length // (2 ** len(hidden_channels))
        self.fc = nn.Linear(latent_dim, hidden_channels[0] * self.init_length)
        self.init_channels = hidden_channels[0]

        layers: list[nn.Module] = []
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(ResBlock1D(in_ch, out_ch))
            in_ch = out_ch

        # Final upsample + projection
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(nn.Conv1d(in_ch, output_channels, kernel_size=3, padding=1))

        self.backbone = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor  — ``[B, latent_dim]``

        Returns
        -------
        x_recon : Tensor  — ``[B, output_channels, output_length]``
        """
        h = self.fc(z)  # [B, C0 * L0]
        h = h.view(h.size(0), self.init_channels, self.init_length)  # [B, C0, L0]
        x_recon = self.backbone(h)
        return x_recon
