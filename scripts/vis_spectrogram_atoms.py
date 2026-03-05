#!/usr/bin/env python
"""
Premium Spectrogram Atom Visualization (FSDD)
Inspired by vis_mnist_atoms.py:
- Sorted by empirical activation probability.
- Heatmap-colored borders reflecting atom importance.
- Red/Blue "Strokes" visualization (Atoms relative to decoder bias).
- Full reconstruction grid (Magma).
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
from torch.utils.data import DataLoader

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize import build_model
from data.datasets import get_fsdd_dataset

def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device)
    except Exception as exc:
        print(f"WARNING: Checkpoint loading failed ({exc}).")
        sys.exit(1)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        state_dict = payload
        
    return state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Dataset Params (Must match training!)
    parser.add_argument("--n-atoms", type=int, default=128)
    parser.add_argument("--input-length", type=int, default=64)
    parser.add_argument("--input-channels", type=int, default=129)
    
    # Spectrogram Params (Must match dataset creation)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=64)
    
    # Model Architecture Overrides
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--encoder-output-dim", type=int, default=256)
    parser.add_argument("--encoder-type", type=str, default="resnet")
    parser.add_argument("--decoder-type", type=str, default="convnmf")
    parser.add_argument("--dict-init", type=str, default="random")
    parser.add_argument("--motif-width", type=int, default=16)
    parser.add_argument("--decoder-stride", type=int, default=16)
    parser.add_argument("--k-max", type=float, default=1.5)
    parser.add_argument("--magnitude-dist", type=str, default="gamma")
    parser.add_argument("--structure-mode", type=str, default="ternary")
    parser.add_argument("--disable-spectrogram-enhancements", action="store_false", dest="spectrogram_enhancements", help="Disable non-negativity and instance bounds.")

    args, _ = parser.parse_known_args()
    args.dataset = 'fsdd' # Force fsdd logic
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading {args.checkpoint}...")
    model = build_model(args)
    
    model.load_state_dict(load_state_dict(Path(args.checkpoint), device))
    model.to(device)
    model.eval()

    n_atoms = model.latent.n_atoms
    print(f"DEBUG: n_atoms={n_atoms}, decoder_weight={model.decoder.weight.shape}")
    
    freq_bins = args.input_channels
    time_frames = args.input_length

    # 2. Empirical Statistics (Sorting)
    print("Collecting empirical activation statistics from FSDD...")
    # Load a small batch of data
    ds = get_fsdd_dataset(data_dir=str(REPO_ROOT / "data" / "fsdd"), 
                          n_fft=args.n_fft, hop_length=args.hop_length, max_frames=args.max_frames,
                          use_instance_norm=args.spectrogram_enhancements)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    batch_x, _ = next(iter(loader))
    batch_x = batch_x.to(device)

    with torch.no_grad():
        _, info = model(batch_x, temp=0.05, sampling="deterministic")
        delta = info.get("delta")
        # Marginal probability of activation: E[|delta|]
        activation_prob = delta.abs().mean(dim=(0, 2)).cpu() # [n_atoms]
        
        # Mean magnitude when active
        gamma = info.get("gamma", torch.ones_like(delta))
        active_mask = delta.abs() > 0.5
        # [Batch, n_atoms, Time]
        mag_when_active = (gamma.abs() * active_mask).sum(dim=(0, 2)) / (active_mask.sum(dim=(0, 2)) + 1e-6)
        mag_when_active = mag_when_active.cpu()

    # 3. One-Hot Decoding & Baseline
    print("Decoding atoms and baseline...")
    with torch.no_grad():
        is_conv = args.decoder_type == "convnmf"
        
        # Baseline (z=0)
        if is_conv:
            T_latent = time_frames // model.decoder.stride
            z_zero = torch.zeros(1, n_atoms, T_latent, device=device)
        else:
            z_zero = torch.zeros(1, model.latent.latent_dim, device=device)
        
        print(f"DEBUG: z_zero shape={z_zero.shape}")
        baseline_flat = model.decoder(z_zero).view(1, -1)
        
        # Individual Atoms (z_i = scale_i)
        scales = torch.where(mag_when_active > 0, mag_when_active, mag_when_active.mean().clamp_min(1.0))
        print(f"DEBUG: scales min={scales.min().item():.4f}, max={scales.max().item():.4f}, mean={scales.mean().item():.4f}")
        z_one_hot = torch.diag(scales).to(device)
        
        if is_conv:
            T_latent = time_frames // model.decoder.stride
            z_conv = torch.zeros(n_atoms, n_atoms, T_latent, device=device)
            # Center the motif in the expanded temporal dimension
            z_conv[:, :, T_latent // 2] = z_one_hot
            z_one_hot = z_conv
        
        if hasattr(model.latent, 'dictionary') and not is_conv:
            z_cont = model.latent.dictionary(z_one_hot)
        else:
            z_cont = z_one_hot
            
        atoms_recon_flat = model.decoder(z_cont).view(n_atoms, -1)
        print(f"DEBUG: recon min={atoms_recon_flat.min().item():.4f}, max={atoms_recon_flat.max().item():.4f}")
        
        # Differential "Strokes"
        atoms_diff_flat = atoms_recon_flat - baseline_flat
        print(f"DEBUG: diff min={atoms_diff_flat.min().item():.4f}, max={atoms_diff_flat.max().item():.4f}")

    # 4. Sorting
    sort_idx = torch.argsort(activation_prob, descending=True)
    activation_prob = activation_prob[sort_idx]
    atoms_recon_flat = atoms_recon_flat[sort_idx]
    atoms_diff_flat = atoms_diff_flat[sort_idx]

    # 5. Reshaping
    actual_frames = atoms_recon_flat.shape[-1] // freq_bins
    atoms_recon_2d = atoms_recon_flat.view(n_atoms, freq_bins, actual_frames).cpu().numpy()
    atoms_diff_2d = atoms_diff_flat.view(n_atoms, freq_bins, actual_frames).cpu().numpy()

    # 6. Premium Plotting
    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        out_path = Path(args.checkpoint).parent / "figures_fsdd_premium"
    out_path.mkdir(parents=True, exist_ok=True)

    def plot_premium_grid(data, cmap, title, filename, vmin=None, vmax=None):
        bg_color = '#1a1a1a'
        text_color = '#eeeeee'
        plt.rcParams.update({'text.color': text_color, 'axes.labelcolor': text_color,
                             'xtick.color': text_color, 'ytick.color': text_color})
        
        n_plot = min(n_atoms, 64)
        cols = 8
        rows = (n_plot + cols - 1) // cols
        
        fig = plt.figure(figsize=(22, 2.5 * rows + 2), facecolor=bg_color)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 15], width_ratios=[40, 1])
        
        # Title
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, title, fontsize=28, weight='bold', ha='center', va='center')
        
        # Main Grid
        ax_main = fig.add_subplot(gs[1, 0])
        ax_main.axis('off')
        
        # Colorbar scale for activation prob
        border_cmap = mpl.colormaps['plasma']
        border_norm = plt.Normalize(vmin=0, vmax=float(activation_prob.max().clamp_min(0.01)))
        
        for idx in range(n_plot):
            r = idx // cols
            c = idx % cols
            
            sub_ax = fig.add_axes([0.05 + c * 0.11, 0.05 + (rows - 1 - r) * 0.8 / rows, 0.10, 0.7 / rows])
            
            img = data[idx]
            sub_ax.imshow(img, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            sub_ax.set_xticks([]); sub_ax.set_yticks([])
            
            prob = float(activation_prob[idx])
            color = border_cmap(border_norm(prob))
            for spine in sub_ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            
            sub_ax.set_title(f"Atom {sort_idx[idx]} (p={prob:.3f})", fontsize=10, pad=2)

        ax_cbar = fig.add_subplot(gs[1, 1])
        sm = plt.cm.ScalarMappable(cmap=border_cmap, norm=border_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cbar)
        cbar.set_label(r'Activation Prob $\mathbb{E}[|\delta|]$', fontsize=18, labelpad=15)
        cbar.outline.set_edgecolor(text_color)
        
        plt.savefig(out_path / filename, bbox_inches='tight', facecolor=bg_color, dpi=120)
        plt.close()
        print(f"✓ Saved {filename}")

    # Plot 1: Full Reconstructions (Magma)
    plot_premium_grid(atoms_recon_2d, 'magma', 
                      "FSDD Learned Atoms: Full Spectrogram Basis", 
                      "fsdd_atoms_full.png")

    # Plot 2: Differential Strokes (BWR)
    # 0 is the neutral center (decoder bias)
    max_abs = np.abs(atoms_diff_2d).max()
    plot_premium_grid(atoms_diff_2d, 'bwr', 
                      "FSDD Learned Strokes: Additive(Red) vs Subtractive(Blue)", 
                      "fsdd_atoms_strokes.png", 
                      vmin=-max_abs, vmax=max_abs)

if __name__ == "__main__":
    main()
