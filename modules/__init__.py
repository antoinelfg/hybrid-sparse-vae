"""Neural architecture modules for the Hybrid Sparse VAE.

Encoder/Decoder layers, Dictionary matrix, and the Structured Latent Space
that implements the polar factorization z = A · (γ ⊙ δ).
"""

from .layers import Encoder1D, Decoder1D, ResBlock1D
from .dictionary import DictionaryMatrix
from .latent_space import StructuredLatentSpace

__all__ = [
    "Encoder1D",
    "Decoder1D",
    "ResBlock1D",
    "DictionaryMatrix",
    "StructuredLatentSpace",
]
