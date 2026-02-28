"""Hybrid Sparse VAE — full generative model.

Combines:
  * Encoder/decoder backbones (linear or ResNet),
  * A structured latent space with polar factorization (γ × δ), and
  * Exact KL-divergence losses (Gamma + Categorical).

Encoder/Decoder modes:
  * ``"linear"`` — symmetric MLP (recommended for toy experiments)
  * ``"resnet"`` — deep Conv1D (for real data)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from modules.layers import Encoder1D, Decoder1D
from modules.latent_space import StructuredLatentSpace


# ---------------------------------------------------------------------------
#  Simple linear encoder/decoder for toy experiments
# ---------------------------------------------------------------------------

class LinearEncoder1D(nn.Module):
    """Simple MLP encoder: input → flatten → Linear → ReLU → Linear → h.

    Symmetric counterpart of :class:`LinearDecoder1D`. For toy problems,
    a symmetric encoder/decoder pair ensures clean gradient flow.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_channels * input_length, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C, T] → h: [B, output_dim]"""
        return self.net(x.view(x.size(0), -1))


class LinearDecoder1D(nn.Module):
    """Deliberately simple decoder: z → Linear → ReLU → Linear → output."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        output_channels: int = 1,
        output_length: int = 128,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.output_length = output_length
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_channels * output_length),
        )

    def forward(self, z: Tensor) -> Tensor:
        out = self.net(z)  # [B, C*T]
        return out.view(z.size(0), self.output_channels, self.output_length)


class HybridSparseVAE(nn.Module):
    """End-to-end Hybrid Sparse VAE.

    Parameters
    ----------
    input_channels, input_length : int
        Signal shape.
    encoder_type, decoder_type : str
        ``"linear"`` (toy) or ``"resnet"`` (real data).
    encoder_output_dim : int
        Feature dimension between encoder and latent space.
    n_atoms, latent_dim : int
        Dictionary width and physical latent dimension.
    dict_init : str
        ``"dct"``, ``"random"``, or ``"identity"``.
    normalize_dict : bool
        L2-normalize dictionary columns.
    k_min : float
        Minimum Gamma shape parameter.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 128,
        encoder_type: str = "linear",
        encoder_output_dim: int = 256,
        n_atoms: int = 64,
        latent_dim: int = 128,
        decoder_type: str = "linear",
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        dict_init: str = "dct",
        normalize_dict: bool = True,
        k_min: float = 0.1,
    ):
        super().__init__()

        # ---- Encoder ---------------------------------------------------
        if encoder_type == "linear":
            self.encoder = LinearEncoder1D(
                input_channels=input_channels,
                input_length=input_length,
                hidden_dim=256,
                output_dim=encoder_output_dim,
            )
        else:
            self.encoder = Encoder1D(
                input_channels=input_channels,
                hidden_channels=encoder_hidden,
                output_dim=encoder_output_dim,
            )

        # ---- Structured Latent Space -----------------------------------
        self.latent = StructuredLatentSpace(
            input_dim=encoder_output_dim,
            n_atoms=n_atoms,
            latent_dim=latent_dim,
            dict_init=dict_init,
            normalize_dict=normalize_dict,
            k_min=k_min,
        )

        # ---- Decoder ---------------------------------------------------
        if decoder_type == "linear":
            self.decoder = LinearDecoder1D(
                latent_dim=latent_dim,
                hidden_dim=256,
                output_channels=input_channels,
                output_length=input_length,
            )
        else:
            self.decoder = Decoder1D(
                latent_dim=latent_dim,
                hidden_channels=decoder_hidden,
                output_channels=input_channels,
                output_length=input_length,
            )

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        temp: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        x : Tensor — ``[B, C, T]``
        temp : float
            Gumbel-Softmax temperature.

        Returns
        -------
        x_recon : Tensor — ``[B, C, T]``
        info : dict
            All diagnostic tensors from the latent space, plus
            ``"x_recon"``.
        """
        h = self.encoder(x)              # [B, encoder_output_dim]
        z, info = self.latent(h, temp=temp)  # [B, latent_dim]
        x_recon = self.decoder(z)         # [B, C, T]

        info["x_recon"] = x_recon
        return x_recon, info
