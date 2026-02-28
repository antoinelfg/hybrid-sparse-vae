"""Encoder and Decoder building blocks (1-D convolutional ResNets).

These are generic backbones for the Hybrid Sparse VAE. The encoder
maps raw input (e.g. spectral frames) to the parameter space consumed
by :class:`modules.latent_space.StructuredLatentSpace`, while the
decoder reconstructs the input from the latent code **z**.
"""

from __future__ import annotations

import torch
import torch.nn as nn
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
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  — ``[B, C, T]``

        Returns
        -------
        h : Tensor  — ``[B, output_dim]``
        """
        h = self.backbone(x)       # [B, last_ch, T']
        h = self.pool(h).squeeze(-1)  # [B, last_ch]
        h = self.fc(h)              # [B, output_dim]
        return h


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
