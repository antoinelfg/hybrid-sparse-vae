#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize import build_model
from data.datasets import get_mnist_dataset


def find_hydra_config(ckpt_path: Path) -> Path | None:
    """Find nearest .hydra/config.yaml from checkpoint parent upward."""
    search_roots = [ckpt_path.parent, *ckpt_path.parents]
    for root in search_roots:
        cfg_path = root / ".hydra" / "config.yaml"
        if cfg_path.exists():
            return cfg_path
        if root == REPO_ROOT:
            break
    return None


def load_hydra_config(ckpt_path: Path) -> tuple[dict[str, Any], Path | None]:
    """Load Hydra config dict if available near the checkpoint."""
    cfg_path = find_hydra_config(ckpt_path)
    if cfg_path is None:
        return {}, None

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"WARNING: Could not parse Hydra config at {cfg_path}: {exc}")
        return {}, None

    if not isinstance(data, dict):
        print(f"WARNING: Hydra config at {cfg_path} is not a dict. Ignoring.")
        return {}, None
    return data, cfg_path


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load plain or wrapped state_dict from checkpoint."""
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except Exception as exc:
        print(f"WARNING: weights_only checkpoint loading failed ({exc}). Retrying full load.")
        payload = torch.load(path, map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")

    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Visualize MNIST Atoms")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Model args (should match training). If omitted, tries .hydra/config.yaml.
    parser.add_argument("--n-atoms", type=int, default=None)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--encoder-output-dim", type=int, default=None)
    parser.add_argument("--encoder-type", type=str, default=None)
    parser.add_argument("--decoder-type", type=str, default=None)
    parser.add_argument("--dict-init", type=str, default=None)
    parser.add_argument("--k-min", type=float, default=None)
    parser.add_argument("--magnitude-dist", type=str, default=None, choices=["gamma", "gaussian"])
    parser.add_argument("--structure-mode", type=str, default=None, choices=["ternary", "binary"])

    # Atom amplitude. If omitted:
    # gamma -> k0 * theta0 from Hydra config, gaussian -> 1.0
    parser.add_argument("--atom-scale", type=float, default=None)
    parser.add_argument("--k-scale", type=float, default=None, help="Alias for --atom-scale.")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    hydra_cfg, hydra_cfg_path = load_hydra_config(ckpt_path)
    if hydra_cfg_path is not None:
        print(f"Using Hydra config defaults from: {hydra_cfg_path}")

    def resolve_arg(arg_name: str, default: Any, hydra_key: str | None = None) -> Any:
        explicit = getattr(args, arg_name)
        if explicit is not None:
            return explicit
        key = hydra_key if hydra_key is not None else arg_name
        return hydra_cfg.get(key, default)

    args.n_atoms = int(resolve_arg("n_atoms", 256))
    args.input_length = int(resolve_arg("input_length", 784))
    args.latent_dim = int(resolve_arg("latent_dim", 64))
    args.encoder_output_dim = int(resolve_arg("encoder_output_dim", 256))
    args.encoder_type = resolve_arg("encoder_type", "linear")
    args.decoder_type = resolve_arg("decoder_type", "linear")
    args.dict_init = resolve_arg("dict_init", "random")
    args.k_min = float(resolve_arg("k_min", 0.1))
    args.magnitude_dist = resolve_arg("magnitude_dist", "gamma")
    args.structure_mode = resolve_arg("structure_mode", "ternary")

    # Force MNIST logic
    args.dataset = "mnist"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading {ckpt_path}...")
    model = build_model(args)
    try:
        model.load_state_dict(load_state_dict(ckpt_path, device))
    except RuntimeError as e:
        print(f"❌ Error loading weights: {e}")
        print("   -> Check if --n-atoms or --input-length match the checkpoint.")
        sys.exit(1)
        
    model.to(device)
    model.eval()

    # 2. Extract Atoms via Empirical One-Hot Decoding
    print("Generating empirical atom statistics from data...")
    n_atoms = model.latent.n_atoms
    
    # We will compute the average activation probability and magnitude 
    # of each atom across a batch of real data.
    from torch.utils.data import DataLoader
    dataset = get_mnist_dataset(data_dir=str(REPO_ROOT / "data"), flatten=True)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    batch = next(iter(loader))
    real_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
    real_batch = real_batch.to(device)
    
    with torch.no_grad():
        _, info = model(real_batch, temp=0.05, sampling="deterministic")
        delta = info.get("delta")
        gamma = info.get("gamma", torch.ones_like(delta))
        
        if delta is not None:
            activation_prob = delta.abs().mean(dim=0).cpu() # [n_atoms]
            active_mask = delta.abs() > 0.5
            mag_when_active = gamma.abs() * active_mask
            sum_mag = mag_when_active.sum(dim=0)
            count_active = active_mask.sum(dim=0)
            
            avg_mag = torch.zeros_like(sum_mag)
            avg_mag[count_active > 0] = sum_mag[count_active > 0] / count_active[count_active > 0].float()
        else:
            activation_prob = torch.ones(n_atoms).cpu()
            avg_mag = torch.ones(n_atoms, device=device)

    # User overrides
    if args.atom_scale is not None or args.k_scale is not None:
        scale = float(args.atom_scale if args.atom_scale is not None else args.k_scale)
        print(f"WARNING: Overriding empirical atom scale with fixed: {scale:.4f}")
        atom_scales = torch.full((n_atoms,), scale, device=device)
    else:
        # Fallback for completely inactive atoms to avoid scale=0 -> producing nothing
        # We use the mean magnitude of ACTIVE atoms, or 1.0 if everything is broken
        mean_active_mag = avg_mag[avg_mag > 0].mean() if (avg_mag > 0).any() else 1.0
        atom_scales = torch.where(avg_mag > 0, avg_mag, mean_active_mag)
        print("Using empirical, per-atom scales based on average magnitude when active.")

    with torch.no_grad():
        # Input latent [n_atoms, n_atoms]
        # Diagonal matrix where entry i,i is the typical scale of atom i
        z_one_hot = torch.diag(atom_scales).to(device)
        
        # Projection (si dictionnaire explicite)
        if hasattr(model.latent, 'dictionary'):
             z_continuous = model.latent.dictionary(z_one_hot)
        else:
             z_continuous = z_one_hot

        # Decoding
        atoms_recon = model.decoder(z_continuous) # [n_atoms, 784]

        # Calcul de la ligne de base (image moyenne) générée par le biais du MLP
        z_zero = torch.zeros(1, model.latent.latent_dim, device=device)
        baseline_img = model.decoder(z_zero)
        
        # Calcul des atomes purs ("strokes") en soustrayant la moyenne
        atoms_diff = atoms_recon - baseline_img

    # Sort atoms by activation probability descending
    sort_idx = torch.argsort(activation_prob, descending=True)
    activation_prob = activation_prob[sort_idx]
    atoms_recon = atoms_recon[sort_idx]
    atoms_diff = atoms_diff[sort_idx]

    # Reshape & Plot Atoms (Standard)
    atoms_img = atoms_recon.view(-1, 1, 28, 28)
    nrow = 16
    padding = 1
    grid = vutils.make_grid(atoms_img, nrow=nrow, normalize=True, scale_each=True, padding=padding)
    
    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        out_path = ckpt_path.parent / "figures_mnist"
    out_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f"Learned Atoms (Full Decoder Output) - {n_atoms} atoms", fontsize=20)
    plt.savefig(out_path / "atoms_grid.png", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved atoms grid to {out_path / 'atoms_grid.png'}")

    # Plot Differential Atoms ("Strokes") with Colored Borders
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib as mpl
    
    atoms_diff_img = atoms_diff.view(-1, 1, 28, 28)
    
    # We don't normalize with make_grid because it shifts the zero-point!
    # Also make_grid outputs 3 channels, we only need 1 to apply a colormap.
    max_abs = atoms_diff_img.abs().max().item()
    grid_diff = vutils.make_grid(atoms_diff_img, nrow=nrow, normalize=False, padding=padding, pad_value=0)
    grid_diff_1c = grid_diff[0] # take first channel [H, W]
    
    # Premium Dark Theme for the plot
    bg_color = '#1a1a1a'
    text_color = '#eeeeee'
    plt.rcParams.update({'text.color': text_color, 'axes.labelcolor': text_color,
                         'xtick.color': text_color, 'ytick.color': text_color})
    
    fig = plt.figure(figsize=(22, 24), facecolor=bg_color)
    
    # Layout with GridSpec to leave room for the title and legend at the top
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 15], width_ratios=[40, 1])
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    ax = fig.add_subplot(gs[1, 0])
    # Colormap 'bwr' applied to 2D tensor securely. 0 is exactly in the middle!
    img = ax.imshow(grid_diff_1c.cpu().numpy(), cmap='bwr', vmin=-max_abs, vmax=max_abs)
    ax.axis('off')
    
    # --- Legends and Titles ---
    ax_title.text(0.5, 0.8, f"Hybrid Sparse VAE: Visualizing Learned MNIST Strokes ({n_atoms} atoms)", 
                  fontsize=32, weight='bold', ha='center', va='center', color=text_color)
    
    #subtitle = (
    #    "Each cell represents an individual atom's contribution (stroke) decoded into the pixel space.\n"
    #    "Atoms are ordered by marginal activation frequency (top-left = most active).\n"
    #    "• Red Pixels: Adds intensity to the baseline shape | • Blue Pixels: Subtracts intensity from the baseline shape\n"
    #    "• Border Color: Marginal probability of the atom being active in the dataset (Heatmap)."
    #)
   # ax_title.text(0.5, 0.2, subtitle, fontsize=18, ha='center', va='center', 
    #              color='#bbbbbb', style='italic', linespacing=1.6)
    
    # Add borders representing activation
    cmap = mpl.colormaps['plasma'] # Updated to avoid deprecation warning
    norm = plt.Normalize(vmin=0, vmax=float(activation_prob.max().clamp_min_(0.1)))
    
    for idx in range(n_atoms):
        row = idx // nrow
        col = idx % nrow
        
        # Coordinates in the grid image
        y = padding + row * (28 + padding)
        x = padding + col * (28 + padding)
        
        prob = float(activation_prob[idx])
        color = cmap(norm(prob))
        
        rect = patches.Rectangle((x - 0.5, y - 0.5), 28, 28, 
                                 linewidth=3, edgecolor=color, facecolor='none', alpha=0.9)
        ax.add_patch(rect)
        
    # Add a colorbar on the right
    ax_cbar = fig.add_subplot(gs[1, 1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label(r'Activation Probability $\mathbb{E}[|\delta|]$', fontsize=20, labelpad=15)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_edgecolor(text_color)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(out_path / "atoms_strokes.png", bbox_inches='tight', facecolor=bg_color, dpi=150)
    plt.close()
    
    # Reset rcParams to default so subsequent plots aren't messed up
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"✓ Saved premium sorted strokes grid to {out_path / 'atoms_strokes.png'}")

    # 3. Reconstruction Check
    print("Generating reconstructions...")
    # Load real data batch
    # Astuce: On n'a pas besoin du dataset complet, juste un batch
    # On utilise le loader du train.py ou une simple création
    from torch.utils.data import DataLoader
    dataset = get_mnist_dataset(data_dir=str(REPO_ROOT / "data"), flatten=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    real_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
    real_batch = real_batch.to(device)
    
    with torch.no_grad():
        # Forward pass complet
        recon_batch, _ = model(real_batch, temp=0.05, sampling="deterministic")
    
    # Reshape
    real_img = real_batch.view(-1, 1, 28, 28)
    recon_img = recon_batch.view(-1, 1, 28, 28)
    
    # Interleave for comparison: Real, Recon, Real, Recon...
    # Stack dim 0: [R1, Rec1, R2, Rec2 ...]
    n_vis = min(16, real_img.shape[0])  # Show up to 16 pairs
    comps = []
    for i in range(n_vis):
        comps.append(real_img[i])
        comps.append(recon_img[i])
    
    comps = torch.stack(comps)
    grid_recon = vutils.make_grid(comps, nrow=8, normalize=True, scale_each=False, padding=2)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(grid_recon.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title("Reconstructions: Real (Left) vs Reconstructed (Right)", fontsize=16)
    plt.axis('off')
    plt.savefig(out_path / "reconstructions.png", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved reconstructions to {out_path / 'reconstructions.png'}")

if __name__ == "__main__":
    main()
