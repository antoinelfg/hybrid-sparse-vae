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
    parser.add_argument("--n-atoms", type=int, default=64)
    parser.add_argument("--input-length", type=int, default=8256)
    
    # Spectrogram Params (Must match dataset creation)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=64)
    
    # Model Architecture Overrides
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--encoder-output-dim", type=int, default=256)
    parser.add_argument("--encoder-type", type=str, default="mlp")
    parser.add_argument("--decoder-type", type=str, default="linear")
    parser.add_argument("--dict-init", type=str, default="random")
    parser.add_argument("--magnitude-dist", type=str, default="gamma")
    parser.add_argument("--structure-mode", type=str, default="ternary")
    parser.add_argument("--disable-spectrogram-enhancements", action="store_false", dest="spectrogram_enhancements", help="Disable non-negativity and instance bounds.")

    args, _ = parser.parse_known_args()
    args.dataset = 'fsdd' # Force fsdd logic
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading {args.checkpoint}...")
    model = build_model(args)
    
    # Enforce Non-negative decoder weights for spectrograms (must match train.py)
    if args.spectrogram_enhancements and args.dataset in ["fsdd", "audio"]:
        import torch.nn.utils.parametrize as parametrize
        
        class NonNegativeWeight(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softplus(x)
                
        for mod in model.decoder.modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)):
                parametrize.register_parametrization(mod, "weight", NonNegativeWeight())

    model.load_state_dict(load_state_dict(Path(args.checkpoint), device))
    model.to(device)
    model.eval()

    n_atoms = model.latent.n_atoms
    freq_bins = args.n_fft // 2 + 1
    time_frames = args.input_length // freq_bins

    # 2. Empirical Statistics (Sorting)
    print("Collecting empirical activation statistics from FSDD...")
    # Load a small batch of data
    ds = get_fsdd_dataset(data_dir=str(REPO_ROOT / "data" / "fsdd"), 
                          n_fft=args.n_fft, hop_length=args.hop_length, max_frames=args.max_frames,
                          use_instance_norm=args.spectrogram_enhancements)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    batch_x, = next(iter(loader))
    batch_x = batch_x.to(device)

    with torch.no_grad():
        _, info = model(batch_x, temp=0.05, sampling="deterministic")
        delta = info.get("delta")
        # Marginal probability of activation: E[|delta|]
        activation_prob = delta.abs().mean(dim=0).cpu() # [n_atoms]
        
        # Mean magnitude when active
        gamma = info.get("gamma", torch.ones_like(delta))
        active_mask = delta.abs() > 0.5
        mag_when_active = (gamma.abs() * active_mask).sum(0) / (active_mask.sum(0) + 1e-6)
        mag_when_active = mag_when_active.cpu()

    # 3. One-Hot Decoding & Baseline
    print("Decoding atoms and baseline...")
    with torch.no_grad():
        # Baseline (z=0)
        z_zero = torch.zeros(1, model.latent.latent_dim, device=device)
        baseline_flat = model.decoder(z_zero).view(1, -1)
        
        # Individual Atoms (z_i = scale_i)
        # We use a scale relative to the empirical mean of the atom to see its "true" form
        scales = torch.where(mag_when_active > 0, mag_when_active, mag_when_active.mean().clamp_min(1.0))
        z_one_hot = torch.diag(scales).to(device)
        
        if hasattr(model.latent, 'dictionary'):
            z_cont = model.latent.dictionary(z_one_hot)
        else:
            z_cont = z_one_hot
            
        atoms_recon_flat = model.decoder(z_cont).view(n_atoms, -1)
        
        # Differential "Strokes"
        atoms_diff_flat = atoms_recon_flat - baseline_flat

    # 4. Sorting
    sort_idx = torch.argsort(activation_prob, descending=True)
    activation_prob = activation_prob[sort_idx]
    atoms_recon_flat = atoms_recon_flat[sort_idx]
    atoms_diff_flat = atoms_diff_flat[sort_idx]

    # 5. Reshaping
    atoms_recon_2d = atoms_recon_flat.view(n_atoms, freq_bins, time_frames).cpu().numpy()
    atoms_diff_2d = atoms_diff_flat.view(n_atoms, freq_bins, time_frames).cpu().numpy()

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
        
        # Iterate and plot sub-images manually to control borders
        # We'll use absolute positions or just a nested gridspec if preferred,
        # but manual calculation is cleaner for custom borders.
        
        for idx in range(n_plot):
            r = idx // cols
            c = idx % cols
            
            # Position for this atom in the figure
            # Subplot axes are easier
            sub_ax = fig.add_axes([0.05 + c * 0.11, 0.05 + (rows - 1 - r) * 0.8 / rows, 0.10, 0.7 / rows])
            
            img = data[idx]
            sub_ax.imshow(img, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            sub_ax.set_xticks([]); sub_ax.set_yticks([])
            
            # Add border
            prob = float(activation_prob[idx])
            color = border_cmap(border_norm(prob))
            for spine in sub_ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            
            sub_ax.set_title(f"Atom {sort_idx[idx]} (p={prob:.2f})", fontsize=9, pad=2)

        # Legend Bar
        ax_cbar = fig.add_subplot(gs[1, 1])
        sm = plt.cm.ScalarMappable(cmap=border_cmap, norm=border_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cbar)
        cbar.set_label('Activation Prob $\mathbb{E}[|\delta|]$', fontsize=18, labelpad=15)
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
