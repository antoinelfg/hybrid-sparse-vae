#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib as mpl
import yaml

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize import build_model
from train import make_toy_data

def find_hydra_config(ckpt_path: Path) -> Path | None:
    search_roots = [ckpt_path.parent, *ckpt_path.parents]
    for root in search_roots:
        cfg_path = root / ".hydra" / "config.yaml"
        if cfg_path.exists():
            return cfg_path
        if root == REPO_ROOT:
            break
    return None

def load_hydra_config(ckpt_path: Path) -> tuple[dict[str, Any], Path | None]:
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
        return {}, None
    return data, cfg_path

def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except Exception as exc:
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


def infer_arch_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    out: dict[str, int] = {}
    dict_w = state_dict.get("latent.dictionary.weight")
    if torch.is_tensor(dict_w) and dict_w.dim() == 2:
        out["latent_dim"] = int(dict_w.shape[0])
        out["n_atoms"] = int(dict_w.shape[1])

    dec_w = state_dict.get("decoder.net.0.weight")
    if torch.is_tensor(dec_w) and dec_w.dim() == 2:
        out["input_length"] = int(dec_w.shape[0])
    return out

def main():
    parser = argparse.ArgumentParser(description="Visualize Sinusoid Atoms")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--n-atoms", type=int, default=None)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--encoder-output-dim", type=int, default=None)
    parser.add_argument("--encoder-type", type=str, default=None)
    parser.add_argument("--decoder-type", type=str, default=None)
    parser.add_argument("--dict-init", type=str, default=None)
    parser.add_argument("--k-min", type=float, default=None)
    parser.add_argument("--magnitude-dist", type=str, default=None)
    parser.add_argument("--structure-mode", type=str, default=None)

    parser.add_argument("--atom-scale", type=float, default=None)

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    hydra_cfg, hydra_cfg_path = load_hydra_config(ckpt_path)
    if hydra_cfg_path is not None:
        print(f"Using Hydra config defaults from: {hydra_cfg_path}")
    state_dict = load_state_dict(ckpt_path, torch.device("cpu"))
    inferred = infer_arch_from_state_dict(state_dict)

    def resolve_arg(arg_name: str, default: Any, hydra_key: str | None = None) -> Any:
        explicit = getattr(args, arg_name)
        if explicit is not None:
            return explicit
        key = hydra_key if hydra_key is not None else arg_name
        if key in hydra_cfg:
            return hydra_cfg[key]
        return inferred.get(arg_name, default)

    args.n_atoms = int(resolve_arg("n_atoms", 32))
    args.input_length = int(resolve_arg("input_length", 128))
    # Provide a safe fallback for latent_dim if it's missing
    args.latent_dim = int(resolve_arg("latent_dim", args.n_atoms // 2))
    args.encoder_output_dim = int(resolve_arg("encoder_output_dim", 256))
    args.encoder_type = resolve_arg("encoder_type", "linear")
    args.decoder_type = resolve_arg("decoder_type", "linear")
    args.dict_init = resolve_arg("dict_init", "random")
    args.k_min = float(resolve_arg("k_min", 10.0))
    args.magnitude_dist = resolve_arg("magnitude_dist", "gamma")
    args.structure_mode = resolve_arg("structure_mode", "ternary")

    args.dataset = "sinusoid"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {ckpt_path}...")
    model = build_model(args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Generating empirical atom statistics from data...")
    from torch.utils.data import DataLoader
    dataset = make_toy_data(n_samples=2048, length=args.input_length)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    batch = next(iter(loader))
    real_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
    real_batch = real_batch.to(device)
    
    with torch.no_grad():
        _, info = model(real_batch, temp=0.05, sampling="deterministic")
        delta = info.get("delta")
        gamma = info.get("gamma", torch.ones_like(delta))
        
        if delta is not None:
            activation_prob = delta.abs().mean(dim=0).squeeze(-1).cpu()
            active_mask = delta.abs() > 0.5
            mag_when_active = gamma.abs() * active_mask
            sum_mag = mag_when_active.sum(dim=0).squeeze(-1)
            count_active = active_mask.sum(dim=0).squeeze(-1)
            
            avg_mag = torch.zeros_like(sum_mag)
            avg_mag[count_active > 0] = sum_mag[count_active > 0] / count_active[count_active > 0].float()
        else:
            activation_prob = torch.ones(args.n_atoms).cpu()
            avg_mag = torch.ones(args.n_atoms, device=device)

    if args.atom_scale is not None:
        scale = float(args.atom_scale)
        atom_scales = torch.full((args.n_atoms,), scale, device=device)
    else:
        mean_active_mag = avg_mag[avg_mag > 0].mean() if (avg_mag > 0).any() else 1.0
        atom_scales = torch.where(avg_mag > 0, avg_mag, mean_active_mag)

    with torch.no_grad():
        z_one_hot = torch.diag(atom_scales).to(device)
        if hasattr(model.latent, 'dictionary'):
             z_continuous = model.latent.dictionary(z_one_hot)
        else:
             z_continuous = z_one_hot

        atoms_recon = model.decoder(z_continuous)
        if atoms_recon.dim() == 2:
            pass
        elif atoms_recon.dim() == 3:
            atoms_recon = atoms_recon.squeeze(1)

        z_zero = torch.zeros(1, model.latent.latent_dim, device=device)
        baseline_img = model.decoder(z_zero)
        if baseline_img.dim() == 3:
            baseline_img = baseline_img.squeeze(1)
        atoms_diff = atoms_recon - baseline_img

    sort_idx = torch.argsort(activation_prob, descending=True)
    activation_prob = activation_prob[sort_idx]
    atoms_recon = atoms_recon[sort_idx]
    atoms_diff = atoms_diff[sort_idx]

    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        out_path = ckpt_path.parent / "figures_sinusoid"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Premium Dark Theme for the plot
    bg_color = '#1a1a1a'
    text_color = '#eeeeee'
    plt.rcParams.update({'text.color': text_color, 'axes.labelcolor': text_color,
                         'xtick.color': text_color, 'ytick.color': text_color,
                         'axes.facecolor': bg_color, 'figure.facecolor': bg_color})

    def plot_curve_grid(curves: torch.Tensor, title: str, filename: str) -> None:
        n_cols = 4
        n_rows = (args.n_atoms + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), sharey=True, sharex=True)
        if args.n_atoms == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        cmap = mpl.colormaps['plasma']
        norm = plt.Normalize(vmin=0, vmax=float(activation_prob.max().clamp_min_(0.01)))

        for i in range(args.n_atoms):
            ax = axes[i]
            prob = float(activation_prob[i])
            color = cmap(norm(prob))

            curve = curves[i].cpu().numpy()
            ax.plot(curve, linewidth=2, color='white')

            ax.set_facecolor(mcolors.to_rgba(color, alpha=0.3))
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

            ax.text(0.05, 0.85, f"P={prob:.2f}", transform=ax.transAxes, color='white',
                    fontsize=10, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        plt.suptitle(title, fontsize=20, weight='bold', color=text_color)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(out_path / filename, bbox_inches='tight', dpi=150)
        plt.close()

    plot_curve_grid(
        atoms_recon,
        f"Hybrid Sparse VAE: Learned 1D Atoms ({args.n_atoms} atoms)",
        "atoms_1d_curves.png",
    )
    plot_curve_grid(
        atoms_diff,
        f"Hybrid Sparse VAE: 1D Atom Strokes ({args.n_atoms} atoms)",
        "atoms_strokes_1d_curves.png",
    )
    
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"✓ Saved 1D curves to {out_path / 'atoms_1d_curves.png'}")
    print(f"✓ Saved 1D strokes to {out_path / 'atoms_strokes_1d_curves.png'}")

    print("Generating reconstructions...")
    plt.rcParams.update({'text.color': text_color, 'axes.labelcolor': text_color,
                         'xtick.color': text_color, 'ytick.color': text_color,
                         'axes.facecolor': bg_color, 'figure.facecolor': bg_color})
                         
    dataset = make_toy_data(n_samples=16, length=args.input_length)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    real_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
    real_batch = real_batch.to(device)
    
    with torch.no_grad():
        recon_batch, _ = model(real_batch, temp=0.05, sampling="deterministic")
    
    real_sig = real_batch.squeeze().cpu().numpy() if real_batch.dim() > 2 else real_batch.cpu().numpy()
    recon_sig = recon_batch.squeeze().cpu().numpy() if recon_batch.dim() > 2 else recon_batch.cpu().numpy()
    
    n_vis = min(8, real_sig.shape[0])
    fig, axes = plt.subplots(n_vis, 1, figsize=(10, 2 * n_vis), sharex=True)
    if n_vis == 1:
        axes = [axes]
    
    for i in range(n_vis):
        axes[i].plot(real_sig[i], label="Real", color="white", alpha=0.9, linestyle="-")
        axes[i].plot(recon_sig[i], label="Recon", color="#ff5555", alpha=0.9, linestyle="--")
        axes[i].legend(loc="upper right")
        
    plt.suptitle("Reconstructions: Real vs Reconstructed", fontsize=16, color=text_color)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(out_path / "reconstructions_1d.png", bbox_inches='tight', dpi=150)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"✓ Saved reconstructions to {out_path / 'reconstructions_1d.png'}")

if __name__ == "__main__":
    main()
