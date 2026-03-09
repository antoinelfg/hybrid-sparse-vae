"""LibriMix-specific experiment models.

This module contains:
  - a simple supervised TF-mask baseline for pipeline calibration
  - a hybrid latent-partition separator that reuses the current
    encoder/latent/shared ConvNMF decoder family but trains with
    source-level supervision
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hybrid_vae import HybridSparseVAE


class DilatedResidualBlock1D(nn.Module):
    """Residual temporal block that preserves sequence length."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.PReLU(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
        )
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class SupervisedTFMaskSeparator(nn.Module):
    """Direct supervised two-speaker TF masking baseline.

    The model predicts two masks over the mixture magnitude. A source-wise
    softmax enforces exact mixture consistency in magnitude space:

        S1_hat + S2_hat = |X|
    """

    def __init__(
        self,
        n_freq_bins: int,
        hidden_channels: int = 384,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_freq_bins = int(n_freq_bins)
        self.stem = nn.Sequential(
            nn.Conv1d(n_freq_bins, hidden_channels, kernel_size=1),
            nn.GroupNorm(1, hidden_channels),
            nn.PReLU(hidden_channels),
        )
        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            blocks.append(
                DilatedResidualBlock1D(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** (i % 6),
                    dropout=dropout,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_channels, 2 * n_freq_bins, kernel_size=1)

    def forward(self, mix_mag: torch.Tensor) -> dict[str, torch.Tensor]:
        if mix_mag.dim() != 3:
            raise ValueError(f"Expected mix_mag shape [B, F, T], got {tuple(mix_mag.shape)}")

        x = torch.log1p(mix_mag.clamp_min(0.0))
        h = self.backbone(self.stem(x))
        mask_logits = self.head(h).view(mix_mag.shape[0], 2, self.n_freq_bins, mix_mag.shape[-1])
        masks = torch.softmax(mask_logits, dim=1)
        source_mags = masks * mix_mag.unsqueeze(1)
        return {
            "source_mags": source_mags,
            "masks": masks,
            "mixture_recon": source_mags.sum(dim=1),
        }


class HybridLatentPartitionSeparator(nn.Module):
    """Hybrid separator with a trainable source partition in latent space.

    This keeps the current encoder, structured latent space, and shared
    ConvNMF decoder. A lightweight assignment head partitions the nonnegative
    atom activations across two sources before decoding.
    """

    def __init__(
        self,
        input_channels: int,
        input_length: int,
        encoder_output_dim: int = 256,
        n_atoms: int = 512,
        latent_dim: int = 64,
        motif_width: int = 16,
        decoder_stride: int = 4,
        encoder_type: str = "resnet",
        dict_init: str = "random",
        normalize_dict: bool = True,
        k_min: float = 0.1,
        k_max: float = 0.8,
        magnitude_dist: str = "gamma",
        structure_mode: str = "binary",
        match_encoder_decoder_stride: bool = True,
        assignment_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if encoder_type == "lista":
            raise ValueError("HybridLatentPartitionSeparator does not support encoder_type='lista'")

        self.backbone = HybridSparseVAE(
            input_channels=input_channels,
            input_length=input_length,
            encoder_type=encoder_type,
            encoder_output_dim=encoder_output_dim,
            n_atoms=n_atoms,
            latent_dim=latent_dim,
            decoder_type="convnmf",
            dict_init=dict_init,
            normalize_dict=normalize_dict,
            k_min=k_min,
            k_max=k_max,
            magnitude_dist=magnitude_dist,
            structure_mode=structure_mode,
            motif_width=motif_width,
            decoder_stride=decoder_stride,
            match_encoder_decoder_stride=match_encoder_decoder_stride,
        )
        self.structure_mode = structure_mode
        assignment_hidden = assignment_hidden or encoder_output_dim
        self.assignment_head = nn.Sequential(
            nn.Conv1d(encoder_output_dim, assignment_hidden, kernel_size=3, padding=1),
            nn.PReLU(assignment_hidden),
            nn.Conv1d(assignment_hidden, 2 * n_atoms, kernel_size=1),
        )

    def _decode_with_length(self, activations: torch.Tensor, output_length: int) -> torch.Tensor:
        decoder = self.backbone.decoder
        if not hasattr(decoder, "output_length"):
            return decoder(activations)

        original_length = int(decoder.output_length)
        decoder.output_length = int(output_length)
        try:
            return decoder(activations)
        finally:
            decoder.output_length = original_length

    def forward(
        self,
        mix_mag: torch.Tensor,
        temp: float = 1.0,
        sampling: str = "stochastic",
    ) -> dict[str, torch.Tensor]:
        if mix_mag.dim() != 3:
            raise ValueError(f"Expected mix_mag shape [B, F, T], got {tuple(mix_mag.shape)}")

        h = self.backbone.encoder(mix_mag)
        _, latent_info = self.backbone.latent(h, temp=temp, sampling=sampling)
        B = latent_info["B"]

        if B.dim() != 3:
            raise ValueError(f"Expected latent activations [B, N, T'], got {tuple(B.shape)}")

        if self.structure_mode == "binary":
            B_nonneg = B.clamp_min(0.0)
        else:
            # Magnitude-domain separation cannot use signed cancellation.
            B_nonneg = B.abs()

        logits = self.assignment_head(h).view(h.shape[0], 2, B_nonneg.shape[1], B_nonneg.shape[2])
        source_assign = torch.softmax(logits, dim=1)
        source_acts = B_nonneg.unsqueeze(1) * source_assign

        source_1 = self._decode_with_length(source_acts[:, 0], output_length=mix_mag.shape[-1])
        source_2 = self._decode_with_length(source_acts[:, 1], output_length=mix_mag.shape[-1])
        source_mags = torch.stack([source_1, source_2], dim=1)
        mixture_recon = source_mags.sum(dim=1)

        out = dict(latent_info)
        out.update(
            {
                "source_assign": source_assign,
                "source_acts": source_acts,
                "source_mags": source_mags,
                "mixture_recon": mixture_recon,
            }
        )
        return out
