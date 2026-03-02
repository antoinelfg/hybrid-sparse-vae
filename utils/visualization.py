"""Visualization utilities for the Hybrid Sparse VAE.

Provides publication-quality plots for:
  * Dictionary atoms (heatmap)
  * Activation matrices (B = γ ⊙ δ)
  * Shape-parameter (k) distributions over training
  * Reconstruction comparison (original vs decoded)
  * Training curves (parsed from log files)
  * Dictionary comparison (DCT vs learned)
  * Sparsity patterns (ternary δ heatmap)
  * Generative samples (prior → decode)
  * Multi-seed summary (bar chart)
"""

from __future__ import annotations

import re
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
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualization utilities.")


def _save(fig: "Figure", save_path: Optional[str | Path]) -> None:
    """Helper to save and close a figure."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  1. Dictionary atom heatmap
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
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
#  2. Activation matrix
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
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
#  3. k-distribution histogram
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
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
#  4. Reconstruction comparison
# ---------------------------------------------------------------------------

def plot_reconstruction_comparison(
    x: Tensor,
    x_recon: Tensor,
    n_examples: int = 5,
    title: str = "Original vs Reconstructed",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """Side-by-side overlay of original vs reconstructed signals.

    Parameters
    ----------
    x : Tensor — ``[B, C, T]`` original signals.
    x_recon : Tensor — ``[B, C, T]`` reconstructed signals.
    n_examples : int — number of examples to show.
    """
    _require_mpl()
    x_np = x.detach().cpu().numpy()
    r_np = x_recon.detach().cpu().numpy()
    n = min(n_examples, x_np.shape[0])

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sig_orig = x_np[i, 0]  # [T]
        sig_reco = r_np[i, 0]
        mse_i = np.mean((sig_orig - sig_reco) ** 2)
        t = np.arange(len(sig_orig))

        ax.plot(t, sig_orig, color="#2196F3", linewidth=1.5, label="Original", alpha=0.9)
        ax.plot(t, sig_reco, color="#F44336", linewidth=1.5, linestyle="--",
                label="Reconstructed", alpha=0.9)
        ax.set_ylabel(f"Sample {i}")
        ax.set_title(f"MSE = {mse_i:.4f}", fontsize=9, loc="right")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time step")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
#  5. Training curves (parsed from log file)
# ---------------------------------------------------------------------------

def parse_train_log(log_path: str | Path) -> dict[str, list]:
    """Parse a train.log file and extract metrics per epoch.

    Expected line format (from train.py logging):
        Epoch  NNN [PHASE] | loss X.XXXX | recon X.XXXX | kl_γ X.XXXX | kl_δ X.XXXX | ...

    Returns dict with keys: epoch, recon, kl_gamma, kl_delta, k_mean, n_active, sparsity, dict_drift
    """
    data = {
        "epoch": [], "recon": [], "kl_gamma": [], "kl_delta": [],
        "k_mean": [], "n_active": [], "sparsity": [], "dict_drift": [],
    }
    path = Path(log_path)
    if not path.exists():
        return data

    pattern = re.compile(
        r"Epoch\s+(\d+).*?"
        r"recon\s+([\d.]+).*?"
        r"kl_γ\s+([\d.]+).*?"
        r"kl_δ\s+([\d.]+).*?"
        r"k̄=([\d.]+).*?"
        r"n_act=([\d.]+).*?"
        r"δ₀=([\d.]+)%.*?"
        r"Δdict=([\d.]+)"
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data["epoch"].append(int(m.group(1)))
                data["recon"].append(float(m.group(2)))
                data["kl_gamma"].append(float(m.group(3)))
                data["kl_delta"].append(float(m.group(4)))
                data["k_mean"].append(float(m.group(5)))
                data["n_active"].append(float(m.group(6)))
                data["sparsity"].append(float(m.group(7)) / 100.0)
                data["dict_drift"].append(float(m.group(8)))

    return data


def plot_training_curves(
    log_path: str | Path,
    phase_boundaries: tuple[int, int, int] = (400, 500, 1000),
    title: str = "Training Curves",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """3-panel training curve from parsed log file.

    Panels: (1) Recon loss, (2) KL-γ, (3) KL-δ.
    Phase boundaries shown as vertical dashed lines with shaded regions.
    """
    _require_mpl()
    data = parse_train_log(log_path)
    if not data["epoch"]:
        raise ValueError(f"No parseable training data found in {log_path}")

    epochs = np.array(data["epoch"])
    p1, p2, p3 = phase_boundaries
    phase_colors = ["#E3F2FD", "#FFF3E0", "#E8F5E9", "#F3E5F5"]  # soft blues/oranges/greens/purples

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    metrics = [
        ("recon", "Reconstruction MSE", "#1976D2"),
        ("kl_gamma", "KL Gamma (γ)", "#E65100"),
        ("kl_delta", "KL Delta (δ)", "#2E7D32"),
    ]

    for ax, (key, label, color) in zip(axes, metrics):
        vals = np.array(data[key])
        ax.plot(epochs, vals, color=color, linewidth=1.2, alpha=0.9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.15)

        # Phase shading
        emin, emax = epochs[0], epochs[-1]
        for start, end, c in [(emin, p1, phase_colors[0]),
                               (p1, p2, phase_colors[1]),
                               (p2, p3, phase_colors[2]),
                               (p3, emax, phase_colors[3])]:
            ax.axvspan(start, end, alpha=0.3, color=c)

        # Phase boundary lines
        for pb in [p1, p2, p3]:
            ax.axvline(x=pb, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    axes[-1].set_xlabel("Epoch", fontsize=11)

    # Phase legend
    phase_labels = ["P1: Soft", "P2: Stochastic", "P3: KL Ramp", "P4: Convergence"]
    patches = [mpatches.Patch(color=c, alpha=0.4, label=l)
               for c, l in zip(phase_colors, phase_labels)]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
#  6. Dictionary comparison (DCT vs learned)
# ---------------------------------------------------------------------------

def plot_dictionary_comparison(
    dct_atoms: Tensor,
    learned_atoms: Tensor,
    title: str = "Dictionary Comparison: DCT vs Learned",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """Side-by-side heatmaps of DCT init vs learned dictionary + similarity histogram.

    Parameters
    ----------
    dct_atoms : Tensor — ``[latent_dim, n_atoms]``
    learned_atoms : Tensor — ``[latent_dim, n_atoms]``
    """
    _require_mpl()
    import torch.nn.functional as Fn

    dct_np = dct_atoms.cpu().numpy()
    lrn_np = learned_atoms.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                              gridspec_kw={"width_ratios": [1, 1, 0.6]})

    # Shared colorbar limits
    vmax = max(np.abs(dct_np).max(), np.abs(lrn_np).max())

    for ax, data, lbl in [(axes[0], dct_np, "DCT Init"),
                           (axes[1], lrn_np, "Learned")]:
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                       interpolation="nearest", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Atom index")
        ax.set_ylabel("Latent dimension")
        ax.set_title(lbl, fontsize=11)

    fig.colorbar(im, ax=axes[:2], shrink=0.8, pad=0.02)

    # Cosine similarity per column (ensure same device)
    cos_sim = Fn.cosine_similarity(dct_atoms.cpu(), learned_atoms.cpu(), dim=0).numpy()
    axes[2].hist(cos_sim, bins=30, color="#7E57C2", edgecolor="white", alpha=0.8)
    axes[2].axvline(x=cos_sim.mean(), color="crimson", linestyle="--",
                    label=f"Mean = {cos_sim.mean():.3f}")
    axes[2].set_xlabel("Cosine Similarity")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Column-wise Similarity")
    axes[2].legend(fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
#  7. Sparsity pattern (ternary δ)
# ---------------------------------------------------------------------------

def plot_sparsity_pattern(
    delta: Tensor,
    n_examples: int = 10,
    title: str = "Sparsity Pattern (δ ∈ {-1, 0, +1})",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """Ternary heatmap of δ patterns with per-atom activation frequency.

    Parameters
    ----------
    delta : Tensor — ``[B, n_atoms]``
    n_examples : int — number of batch examples to show.
    """
    _require_mpl()
    from matplotlib.colors import ListedColormap

    d = delta.detach().cpu().numpy()
    n = min(n_examples, d.shape[0])
    d_sub = d[:n]  # [n, n_atoms]
    n_atoms = d_sub.shape[1]

    # Activation frequency (across full batch)
    freq = (d != 0).astype(float).mean(axis=0)  # [n_atoms]

    fig, (ax_heat, ax_freq) = plt.subplots(
        1, 2, figsize=(14, max(4, 0.08 * n_atoms)),
        gridspec_kw={"width_ratios": [4, 1]}, sharey=True
    )

    # Custom colormap: red=-1, white=0, blue=+1
    # Transpose: atoms on y-axis, samples on x-axis
    cmap = ListedColormap(["#E53935", "#FAFAFA", "#1E88E5"])
    im = ax_heat.imshow(d_sub.T, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                         interpolation="nearest")
    ax_heat.set_xlabel("Sample")
    ax_heat.set_ylabel("Atom index")
    ax_heat.set_title("δ values")

    # Activation frequency horizontal bar (y = atom index, aligned with heatmap)
    ax_freq.barh(np.arange(n_atoms), freq, color="#546E7A", alpha=0.8, height=0.8)
    ax_freq.set_xlabel("Activation freq.")
    ax_freq.set_title("Per-atom freq.")

    # Discrete colorbar
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.6, ticks=[-1, 0, 1])
    cbar.set_ticklabels(["-1", "0", "+1"])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
#  8. Generative samples (prior → decode)
# ---------------------------------------------------------------------------

def plot_generative_samples(
    model,
    n_samples: int = 10,
    k_0: float = 0.3,
    theta_0: float = 1.0,
    delta_prior: tuple[float, float, float] = (0.15, 0.70, 0.15),
    device: str = "cuda",
    title: str = "Generated Samples (Prior → Decode)",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """Sample from the prior and decode, then plot generated signals.

    Parameters
    ----------
    model : HybridSparseVAE (must be in eval mode).
    """
    _require_mpl()
    import torch.distributions as dist

    model.eval()
    n_atoms = model.latent.n_atoms
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # Sample gamma from prior Gamma(k_0, theta_0)
        gamma = dist.Gamma(
            torch.full((n_samples, n_atoms), k_0, device=dev),
            torch.full((n_samples, n_atoms), 1.0 / theta_0, device=dev),  # rate = 1/scale
        ).sample()

        # Sample delta from Categorical prior
        probs = torch.tensor(delta_prior, device=dev)
        cat_idx = dist.Categorical(probs=probs).sample((n_samples, n_atoms))
        delta_map = torch.tensor([-1.0, 0.0, 1.0], device=dev)
        delta = delta_map[cat_idx]

        # B = gamma * delta, then dictionary projection + decode
        B = gamma * delta
        z = model.latent.dictionary(B)
        x_gen = model.decoder(z)  # [n_samples, 1, T]

    x_np = x_gen.cpu().numpy()
    n_cols = 2
    n_rows = (n_samples + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows), sharex=True)
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].plot(x_np[i, 0], color="#5C6BC0", linewidth=1.2)
        axes[i].set_title(f"Sample {i+1}", fontsize=9)
        axes[i].grid(True, alpha=0.2)
        axes[i].set_ylabel("Amplitude", fontsize=7)

    # Hide empty subplots
    for j in range(n_samples, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
#  9. Multi-seed summary
# ---------------------------------------------------------------------------

def plot_multiseed_summary(
    results: dict[int, float],
    metric_name: str = "Recon MSE",
    title: str = "Multi-Seed Results",
    save_path: Optional[str | Path] = None,
) -> "Figure":
    """Bar chart of final metric per seed, color-coded by quality.

    Parameters
    ----------
    results : dict mapping seed → metric value (e.g. {42: 3.47, 123: 1.72, ...}).
    """
    _require_mpl()
    seeds = sorted(results.keys())
    values = [results[s] for s in seeds]
    median_val = float(np.median(values))

    # Color-code: green < 1.5, yellow 1.5-2.5, red > 2.5
    colors = []
    for v in values:
        if v < 1.5:
            colors.append("#43A047")
        elif v < 2.5:
            colors.append("#FFA000")
        else:
            colors.append("#E53935")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seeds))
    bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.6)

    # Value labels on bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Median line
    ax.axhline(y=median_val, color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Median = {median_val:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds], fontsize=9)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
