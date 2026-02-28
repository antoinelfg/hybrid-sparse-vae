"""Visualization utilities for the Hybrid Sparse VAE.

Provides publication-quality plots for:
  * Dictionary atoms (heatmap)
  * Activation matrices (B = γ ⊙ δ)
  * Shape-parameter (k) distributions over training
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch import Tensor

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualization utilities.")


# ---------------------------------------------------------------------------
#  Dictionary atom heatmap
# ---------------------------------------------------------------------------

def plot_atoms(
    atoms: Tensor,
    title: str = "Dictionary Atoms",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """Heatmap of the dictionary matrix **A** [latent_dim × n_atoms].

    Parameters
    ----------
    atoms : Tensor — ``[latent_dim, n_atoms]``
    title : str
    save_path : path, optional
        If given, saves the figure as PNG.
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        atoms.cpu().numpy(),
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax.set_xlabel("Atom index")
    ax.set_ylabel("Latent dimension")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
#  Activation matrix
# ---------------------------------------------------------------------------

def plot_activations(
    B: Tensor,
    title: str = "Activation Matrix  B = γ ⊙ δ",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """Heatmap of the signed sparse activations **B**.

    Parameters
    ----------
    B : Tensor — ``[T, n_atoms]`` (single example) or selects first
        batch element from ``[B, T, n_atoms]``.
    """
    _require_mpl()
    if B.ndim == 3:
        B = B[0]
    data = B.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        data.T,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Atom index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
#  k-distribution histogram
# ---------------------------------------------------------------------------

def plot_k_distribution(
    k: Tensor,
    title: str = "Shape Parameter k Distribution",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """Histogram of predicted shape parameters **k**.

    Useful to verify that k converges below 1 (sparse regime).

    Parameters
    ----------
    k : Tensor — arbitrary shape, will be flattened.
    """
    _require_mpl()
    data = k.detach().cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(x=1.0, color="crimson", linestyle="--", linewidth=1.5, label="k = 1")
    ax.set_xlabel("k (shape parameter)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
