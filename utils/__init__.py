"""Utility functions (metrics, objectives, visualization)."""

from .objectives import compute_hybrid_loss, kl_gamma, kl_categorical
from .visualization import (
    plot_atoms,
    plot_activations,
    plot_k_distribution,
    plot_reconstruction_comparison,
    plot_training_curves,
    plot_dictionary_comparison,
    plot_sparsity_pattern,
    plot_generative_samples,
    plot_multiseed_summary,
    parse_train_log,
)

__all__ = [
    "compute_hybrid_loss", "kl_gamma", "kl_categorical",
    "plot_atoms", "plot_activations", "plot_k_distribution",
    "plot_reconstruction_comparison", "plot_training_curves",
    "plot_dictionary_comparison", "plot_sparsity_pattern",
    "plot_generative_samples", "plot_multiseed_summary",
    "parse_train_log",
]
