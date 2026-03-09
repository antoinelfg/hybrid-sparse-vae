"""Hybrid Sparse VAE — full generative model.

Combines:
  * Encoder/decoder backbones (linear, ResNet, or ConvLISTA),
  * A structured latent space with polar factorization (γ × δ), and
  * Exact KL-divergence losses (Gamma + Categorical).

Encoder/Decoder modes:
  * ``"linear"`` — symmetric MLP (recommended for toy experiments)
  * ``"resnet"`` — deep Conv1D (for real data)
  * ``"lista"`` — Unrolled ISTA inference encoder (convolutional in temporal
                  mode, linear in dense mode)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.layers import (
    Encoder1D,
    Decoder1D,
    ConvUnrolledISTAEncoder,
    LinearUnrolledISTAEncoder,
    PolarLinearLISTAEncoder,
    FullyPolarLinearLISTAEncoder,
)
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
    """Truly strictly linear decoder: z → Linear → output."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,  # Ignored for strict linearity, kept for signature compat
        output_channels: int = 1,
        output_length: int = 128,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.output_length = output_length
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_channels * output_length),
        )

    def forward(self, z: Tensor) -> Tensor:
        out = self.net(z)  # [B, C*T]
        return out.view(z.size(0), self.output_channels, self.output_length)


class NonNegativeLinearDecoder(nn.Module):
    """Truly strictly linear decoder with non-negative weights for NMF.
    
    Weights are forced to be positive via softplus and bias is disabled
    to ensure that zero latent activation yields zero physical output.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,  # Ignored for strict linearity, kept for signature compat
        output_channels: int = 1,
        output_length: int = 128,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.output_length = output_length
        # No bias for pure NMF
        self.net = nn.Linear(latent_dim, output_channels * output_length, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        # Dynamically force weights to be positive
        W_pos = torch.nn.functional.softplus(self.net.weight)
        out = torch.nn.functional.linear(z, W_pos)  # [B, C*T]
        return out.view(z.size(0), self.output_channels, self.output_length)


class ShiftInvariantDecoder(nn.Module):
    """Convolutional NMF (ConvNMF) Decoder.
    
    The dictionary A is the weight of a ConvTranspose1d layer.
    W >= 0 is enforced by passing the weights through a softplus or absolute value.
    """
    
    def __init__(self, n_atoms: int, n_freq_bins: int, motif_width: int = 16, output_length: int = 128, stride: int = 16):
        super().__init__()
        self.output_channels = n_freq_bins
        self.stride = stride
        self.motif_width = motif_width
        self.output_length = output_length
        
        # On déclare les poids comme un Parameter classique.
        # Shape: [in_channels (n_atoms), out_channels (output_channels), kernel_size]
        self.weight = nn.Parameter(torch.Tensor(n_atoms, self.output_channels, motif_width))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : Tensor [Batch, n_atoms, Time]
        """
        # 1. Contrainte Physique NMF stricte (différentiable et propre)
        W_pos = torch.abs(self.weight)  # ou F.softplus(self.weight)
        
        # 2. Convolution Transposée Fonctionnelle
        # Le stride=16 va "espacer" les points temporels et le kernel (motif_width) va remplir l'espace.
        x_recon = F.conv_transpose1d(
            z, 
            weight=W_pos, 
            bias=None, 
            stride=self.stride, 
            padding=self.motif_width // 2
        )
        
        # 3. Ajustement millimétrique (si besoin) au lieu d'une interpolation sauvage
        # Si la taille dépasse légèrement 128 à cause des arrondis de padding, on coupe proprement.
        if x_recon.shape[-1] > self.output_length:
            x_recon = x_recon[..., :self.output_length]
        elif x_recon.shape[-1] < self.output_length:
            # If it's shorter, pad it
            x_recon = F.pad(x_recon, (0, self.output_length - x_recon.shape[-1]))
            
        return x_recon


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
        k_max: float = float("inf"),
        magnitude_dist: str = "gamma",
        structure_mode: str = "ternary",
        motif_width: int = 16, # Added for convnmf
        decoder_stride: int = 16,  # ConvTranspose1d stride for ShiftInvariantDecoder
        match_encoder_decoder_stride: bool = False,
        lista_iterations: int = 5,
        lista_threshold_init: float = 0.1,
        delta_head_mode: str = "shared",
        polar_encoder: bool = False,
        fully_polar_encoder: bool = False,
        shape_norm: str = "l2_global",
        gain_feature: str = "log_l2",
        gamma_scale_injection: str = "multiply_input_norm",
        shape_detach_to_gamma: bool = True,
        delta_factorization: str = "ternary_direct",
        presence_estimator: str = "gumbel_binary",
        sign_estimator: str = "gumbel_binary",
        presence_alpha: float = 1.5,
        tau_presence_eval: float = 0.5,
        sign_tau_eval: float = 0.5,
        presence_head_bias_init: float = 0.0,
        sign_head_bias_init: float = 0.0,
        gumbel_epsilon: float = 0.05,
    ):
        super().__init__()
        
        self.temporal_mode = (decoder_type == "convnmf")
        self.match_encoder_decoder_stride = bool(match_encoder_decoder_stride)
        spatial_pooling = not self.temporal_mode

        # ---- Encoder ---------------------------------------------------
        self.lista_mode = (encoder_type == "lista")
        self.polar_lista_mode = (encoder_type == "polar_lista") or bool(polar_encoder)
        self.fully_polar_lista_mode = (encoder_type == "fully_polar_lista") or bool(fully_polar_encoder)
        if self.polar_lista_mode and self.fully_polar_lista_mode:
            raise ValueError("Choose exactly one of encoder_type='polar_lista' or 'fully_polar_lista'.")
        if self.polar_lista_mode and encoder_type != "polar_lista":
            raise ValueError("Set encoder_type='polar_lista' when polar_encoder=True.")
        if self.fully_polar_lista_mode and encoder_type != "fully_polar_lista":
            raise ValueError("Set encoder_type='fully_polar_lista' when fully_polar_encoder=True.")
        if self.polar_lista_mode and self.temporal_mode:
            raise ValueError("Polar LISTA is only implemented for non-temporal dense runs.")
        if self.fully_polar_lista_mode and self.temporal_mode:
            raise ValueError("Fully Polar LISTA is only implemented for non-temporal dense runs.")
        if delta_factorization != "ternary_direct" and not self.polar_lista_mode:
            if not self.fully_polar_lista_mode:
                raise ValueError("delta_factorization='presence_sign' is only supported with polar LISTA encoders.")
        if self.fully_polar_lista_mode and delta_factorization != "presence_sign":
            raise ValueError("fully_polar_lista requires delta_factorization='presence_sign'.")
        if self.fully_polar_lista_mode and gamma_scale_injection != "multiply_input_norm":
            raise ValueError("fully_polar_lista currently only supports gamma_scale_injection='multiply_input_norm'.")
        if self.fully_polar_lista_mode and sign_estimator != "gumbel_binary":
            raise ValueError("fully_polar_lista currently supports sign_estimator='gumbel_binary' only.")
        if self.fully_polar_lista_mode and self.temporal_mode:
            raise ValueError("fully_polar_lista is not supported with decoder_type='convnmf'.")
        if delta_factorization != "ternary_direct" and not self.polar_lista_mode and not self.fully_polar_lista_mode:
            raise ValueError("delta_factorization='presence_sign' is only supported with encoder_type='polar_lista'.")
        if encoder_type == "linear":
            self.encoder = LinearEncoder1D(
                input_channels=input_channels,
                input_length=input_length,
                hidden_dim=256,
                output_dim=encoder_output_dim,
            )
        elif encoder_type == "lista":
            if self.temporal_mode:
                self.encoder = ConvUnrolledISTAEncoder(
                    input_channels=input_channels,
                    n_atoms=n_atoms,
                    n_iterations=lista_iterations,
                    kernel_size=motif_width,
                    structure_mode=structure_mode,
                    k_max=k_max,
                    delta_head_mode=delta_head_mode,
                )
            else:
                self.encoder = LinearUnrolledISTAEncoder(
                    input_channels=input_channels,
                    input_length=input_length,
                    n_atoms=n_atoms,
                    n_iterations=lista_iterations,
                    structure_mode=structure_mode,
                    k_max=k_max,
                    threshold_init=lista_threshold_init,
                    delta_head_mode=delta_head_mode,
                )
        elif self.polar_lista_mode:
            self.encoder = PolarLinearLISTAEncoder(
                input_channels=input_channels,
                input_length=input_length,
                n_atoms=n_atoms,
                n_iterations=lista_iterations,
                k_min=k_min,
                k_max=k_max,
                shape_norm=shape_norm,
                gain_feature=gain_feature,
                shape_detach_to_gamma=shape_detach_to_gamma,
                presence_head_bias_init=presence_head_bias_init,
                sign_head_bias_init=sign_head_bias_init,
                threshold_init=lista_threshold_init,
            )
        elif self.fully_polar_lista_mode:
            self.encoder = FullyPolarLinearLISTAEncoder(
                input_channels=input_channels,
                input_length=input_length,
                n_atoms=n_atoms,
                n_iterations=lista_iterations,
                k_min=k_min,
                k_max=k_max,
                shape_detach_to_gamma=shape_detach_to_gamma,
                presence_head_bias_init=presence_head_bias_init,
                sign_head_bias_init=sign_head_bias_init,
                threshold_init=lista_threshold_init,
            )
        else:
            # Opt-in architecture-safe mode for ConvNMF runs. Legacy checkpoints
            # keep the historical x16 encoder reduction unless this flag is set.
            temporal_downsample_factor = None
            if self.temporal_mode and self.match_encoder_decoder_stride:
                temporal_downsample_factor = decoder_stride
            self.encoder = Encoder1D(
                input_channels=input_channels,
                hidden_channels=encoder_hidden,
                output_dim=encoder_output_dim,
                spatial_pooling=spatial_pooling,
                temporal_downsample_factor=temporal_downsample_factor,
            )

        # ---- Structured Latent Space -----------------------------------
        self.latent = StructuredLatentSpace(
            input_dim=encoder_output_dim,
            n_atoms=n_atoms,
            latent_dim=latent_dim,
            dict_init=dict_init,
            normalize_dict=normalize_dict,
            k_min=k_min,
            k_max=k_max,
            magnitude_dist=magnitude_dist,
            structure_mode=structure_mode,
            temporal_mode=self.temporal_mode,
            delta_factorization=delta_factorization,
            presence_estimator=presence_estimator,
            sign_estimator=sign_estimator,
            presence_alpha=presence_alpha,
            tau_presence_eval=tau_presence_eval,
            gumbel_epsilon=gumbel_epsilon,
        )

        # ---- Decoder ---------------------------------------------------
        if decoder_type == "linear":
            self.decoder = LinearDecoder1D(
                latent_dim=latent_dim,
                hidden_dim=256,
                output_channels=input_channels,
                output_length=input_length,
            )
        elif decoder_type == "linear_positive":
            self.decoder = NonNegativeLinearDecoder(
                latent_dim=latent_dim,
                hidden_dim=256,
                output_channels=input_channels,
                output_length=input_length,
            )
        elif decoder_type == "convnmf":
            self.decoder = ShiftInvariantDecoder(
                n_atoms=n_atoms,
                n_freq_bins=input_channels,
                motif_width=motif_width,
                output_length=input_length,
                stride=decoder_stride,
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
        sampling: str = "stochastic",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.fully_polar_lista_mode:
            k, theta_tilde, input_scale, presence_logits, sign_logits, aux = self.encoder(x)
            z, info = self.latent.forward_from_fully_factorized_params(
                k,
                theta_tilde,
                input_scale,
                presence_logits,
                sign_logits,
                temp=temp,
                sampling=sampling,
            )
            info.update(aux)
        elif self.polar_lista_mode:
            k, theta, presence_logits, sign_scores, aux = self.encoder(x)
            z, info = self.latent.forward_from_factorized_params(
                k,
                theta,
                presence_logits,
                sign_scores,
                temp=temp,
                sampling=sampling,
            )
            info.update(aux)
        elif self.lista_mode:
            # LISTA encoder returns (k, theta, logits) directly — skip conv_params head
            k, theta, logits = self.encoder(x)
            z, info = self.latent.forward_from_params(k, theta, logits, temp=temp, sampling=sampling)
        else:
            h = self.encoder(x)
            z, info = self.latent(h, temp=temp, sampling=sampling)
        x_recon = self.decoder(z)

        info["x_recon"] = x_recon
        return x_recon, info

    def get_dict_atoms(self) -> torch.Tensor:
        """Returns the dictionary weights safely across architectural variations."""
        if self.temporal_mode:
            return self.decoder.weight.data
        else:
            return self.latent.dictionary.get_atoms()
