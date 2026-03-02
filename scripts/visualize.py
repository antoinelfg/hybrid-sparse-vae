#!/usr/bin/env python
"""CLI script to generate all visualization figures from a trained checkpoint.

Usage
-----
    # From a checkpoint:
    python scripts/visualize.py --checkpoint checkpoints/hybrid_vae_final.pt

    # From a results directory (looks for checkpoint + train.log):
    python scripts/visualize.py --results-dir results/champion_v2_20260228_0954/

    # Override model config:
    python scripts/visualize.py --checkpoint ckpt.pt --dict-init random --n-atoms 128

    # Specify output directory:
    python scripts/visualize.py --checkpoint ckpt.pt --output-dir results/my_run/figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from models.hybrid_vae import HybridSparseVAE
from modules.dictionary import DictionaryMatrix
from train import make_toy_data
from utils.visualization import (
    plot_atoms,
    plot_activations,
    plot_k_distribution,
    plot_reconstruction_comparison,
    plot_training_curves,
    plot_dictionary_comparison,
    plot_sparsity_pattern,
    plot_generative_samples,
    plot_multiseed_summary,
)


def build_model(args) -> HybridSparseVAE:
    """Build model with configuration from args."""
    return HybridSparseVAE(
        input_channels=1,
        input_length=args.input_length,
        encoder_type=args.encoder_type,
        encoder_output_dim=args.encoder_output_dim,
        n_atoms=args.n_atoms,
        latent_dim=args.latent_dim,
        decoder_type=args.decoder_type,
        dict_init=args.dict_init,
        normalize_dict=getattr(args, "normalize_dict", True),
        k_min=getattr(args, "k_min", 0.1),
        magnitude_dist=getattr(args, "magnitude_dist", "gamma"),
        structure_mode=getattr(args, "structure_mode", "ternary"),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory (looks for checkpoint + train.log)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: <results-dir>/figures/)")

    # Model config overrides
    parser.add_argument("--dict-init", type=str, default="dct",
                        choices=["dct", "random", "identity"])
    parser.add_argument("--n-atoms", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--encoder-output-dim", type=int, default=256)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--encoder-type", type=str, default="linear")
    parser.add_argument("--decoder-type", type=str, default="linear")

    # Data config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-examples", type=int, default=5,
                        help="Number of examples for reconstruction plots")

    # Prior config (for generative sampling)
    parser.add_argument("--k0", type=float, default=0.3)
    parser.add_argument("--theta0", type=float, default=1.0)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Resolve checkpoint and output paths
    ckpt_path = None
    log_path = None

    if args.results_dir:
        rdir = Path(args.results_dir)
        # Look for checkpoint in results dir, then checkpoints/, then repo root
        search_dirs = [rdir, REPO_ROOT / "checkpoints", REPO_ROOT]
        ckpt_names = ["hybrid_vae_final.pt", "checkpoint.pt", "model.pt"]
        for sdir in search_dirs:
            for name in ckpt_names:
                candidate = sdir / name
                if candidate.exists():
                    ckpt_path = candidate
                    break
            if ckpt_path:
                break
        # Look for train.log in results dir
        for name in ["train.log"]:
            candidate = rdir / name
            if candidate.exists():
                log_path = candidate
                break
        if args.output_dir is None:
            args.output_dir = str(rdir / "figures")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)

    if ckpt_path is None:
        searched = ", ".join(str(d) for d in [Path(args.results_dir)] if args.results_dir)
        print(f"ERROR: No checkpoint (.pt) found.")
        print(f"  Searched: {searched or 'nowhere'}, checkpoints/, repo root")
        print(f"  Provide --checkpoint explicitly or place a .pt file in the results dir.")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = str(ckpt_path.parent / "figures")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Checkpoint : {ckpt_path}")
    print(f"Log file   : {log_path or 'not found'}")
    print(f"Output dir : {out_dir}")
    print(f"Device     : {device}")
    print()

    # ---- Load model --------------------------------------------------------
    model = build_model(args)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✓ Model loaded")

    # ---- Generate data (same as training) ----------------------------------
    dataset = make_toy_data(
        n_samples=args.n_samples,
        length=args.input_length,
        n_components=3,
        seed=args.seed,
    )
    X = dataset.tensors[0].to(device)  # [N, 1, T]
    print(f"✓ Data: {X.shape}")

    # ---- Forward pass ------------------------------------------------------
    with torch.no_grad():
        x_batch = X[:64]
        x_recon, info = model(x_batch, temp=0.05, sampling="deterministic")
    print("✓ Forward pass complete")

    # ---- 1. Dictionary atoms -----------------------------------------------
    atoms = model.latent.dictionary.get_atoms()
    plot_atoms(atoms, title="Learned Dictionary Atoms",
               save_path=out_dir / "01_dictionary_atoms.png")
    print("✓ [1/8] Dictionary atoms heatmap")

    # ---- 2. Activation matrix -----------------------------------------------
    plot_activations(info["B"][:1], title="Activation Matrix B = γ ⊙ δ",
                     save_path=out_dir / "02_activation_matrix.png")
    print("✓ [2/8] Activation matrix")

    # ---- 3. k-distribution --------------------------------------------------
    plot_k_distribution(info["k"], title="Shape Parameter k Distribution",
                        save_path=out_dir / "03_k_distribution.png")
    print("✓ [3/8] k-distribution")

    # ---- 4. Reconstruction comparison ----------------------------------------
    plot_reconstruction_comparison(
        x_batch[:args.n_examples], x_recon[:args.n_examples],
        n_examples=args.n_examples,
        title="Original vs Reconstructed Signals",
        save_path=out_dir / "04_reconstruction_comparison.png",
    )
    print("✓ [4/8] Reconstruction comparison")

    # ---- 5. Training curves ---------------------------------------------------
    if log_path and log_path.exists():
        try:
            plot_training_curves(
                log_path,
                title="Training Curves",
                save_path=out_dir / "05_training_curves.png",
            )
            print("✓ [5/8] Training curves")
        except ValueError as e:
            print(f"⚠ [5/8] Training curves skipped: {e}")
    else:
        print("⚠ [5/8] Training curves skipped (no train.log found)")

    # ---- 6. Dictionary comparison (DCT vs learned) ---------------------------
    dct_ref = DictionaryMatrix(
        n_atoms=args.n_atoms, latent_dim=args.latent_dim,
        normalize=True, init="dct"
    )
    dct_atoms = dct_ref.get_atoms()
    plot_dictionary_comparison(
        dct_atoms, atoms,
        title="Dictionary Comparison: DCT Init vs Learned",
        save_path=out_dir / "06_dictionary_comparison.png",
    )
    print("✓ [6/8] Dictionary comparison")

    # ---- 7. Sparsity pattern --------------------------------------------------
    plot_sparsity_pattern(
        info["delta"],
        n_examples=min(20, x_batch.shape[0]),
        title="Sparsity Pattern (δ ∈ {-1, 0, +1})",
        save_path=out_dir / "07_sparsity_pattern.png",
    )
    print("✓ [7/8] Sparsity pattern")

    # ---- 8. Generative samples ------------------------------------------------
    plot_generative_samples(
        model, n_samples=10,
        k_0=args.k0, theta_0=args.theta0,
        device=str(device),
        title="Generated Samples (Prior → Decode)",
        save_path=out_dir / "08_generative_samples.png",
    )
    print("✓ [8/8] Generative samples")

    print(f"\n{'='*60}")
    print(f"All figures saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
