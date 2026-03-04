"""Encoder and Decoder building blocks (1-D convolutional ResNets).

These are generic backbones for the Hybrid Sparse VAE. The encoder
maps raw input (e.g. spectral frames) to the parameter space consumed
by :class:`modules.latent_space.StructuredLatentSpace`, while the
decoder reconstructs the input from the latent code **z**.
"""

from __future__ import annotations

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
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: list[int] | None = None,
        output_dim: int = 256,
        spatial_pooling: bool = False,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in hidden_channels:
            layers.append(ResBlock1D(in_ch, out_ch, stride=2))
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)
        self.spatial_pooling = spatial_pooling
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
    ):
        super().__init__()
        self.n_iterations = n_iterations
        self.n_atoms = n_atoms
        self.n_classes = 3 if structure_mode == "ternary" else 2

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
        theta_out = F.softplus(self.head_theta(h)) + 1e-4  # [B, n_atoms, T']

        # 4. Logits: [B, n_atoms * n_classes, T'] → [B, n_atoms, T', n_classes]
        T_prime = h.size(-1)
        logits = self.head_pi(h).view(B, self.n_atoms, self.n_classes, T_prime)
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
