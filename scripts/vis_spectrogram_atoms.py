#!/usr/bin/env python
"""Visualize ConvNMF spectrogram atoms and stroke-like residuals.

Primarily intended for FSDD checkpoints, with configuration loaded from the
nearest Hydra config when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import get_fsdd_dataset
from scripts.visualize import build_model


def find_hydra_config(ckpt_path: Path) -> Path | None:
    for root in [ckpt_path.parent, *ckpt_path.parents]:
        candidate = root / ".hydra" / "config.yaml"
        if candidate.exists():
            return candidate
        if root == REPO_ROOT:
            break
    return None


def load_hydra_config(ckpt_path: Path) -> tuple[dict[str, Any], Path | None]:
    cfg_path = find_hydra_config(ckpt_path)
    if cfg_path is None:
        return {}, None
    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:
        print(f"WARNING: Could not parse Hydra config at {cfg_path}: {exc}")
        return {}, None
    return data if isinstance(data, dict) else {}, cfg_path


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except Exception:
        payload = torch.load(path, map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    if all(key.startswith("model.") for key in state_dict):
        state_dict = {key[len("model."):]: value for key, value in state_dict.items()}
    return state_dict


def infer_arch_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    inferred: dict[str, int] = {}
    decoder_weight = state_dict.get("decoder.weight")
    if torch.is_tensor(decoder_weight) and decoder_weight.dim() == 3:
        inferred["n_atoms"] = int(decoder_weight.shape[0])
        inferred["input_channels"] = int(decoder_weight.shape[1])
        inferred["motif_width"] = int(decoder_weight.shape[2])
    dict_weight = state_dict.get("latent.dictionary.weight")
    if torch.is_tensor(dict_weight) and dict_weight.dim() == 2:
        inferred["latent_dim"] = int(dict_weight.shape[0])
        inferred["n_atoms"] = int(dict_weight.shape[1])
    conv_params_weight = state_dict.get("latent.conv_params.weight")
    if torch.is_tensor(conv_params_weight) and "n_atoms" in inferred and inferred["n_atoms"] > 0:
        ratio = int(conv_params_weight.shape[0] // inferred["n_atoms"])
        if ratio == 4:
            inferred["structure_mode"] = "binary"
        elif ratio == 5:
            inferred["structure_mode"] = "ternary"
    return inferred


def resolve_arg(args: argparse.Namespace, hydra_cfg: dict[str, Any], inferred: dict[str, int], name: str, default: Any) -> Any:
    explicit = getattr(args, name)
    if explicit is not None:
        return explicit
    if name in inferred:
        return inferred[name]
    if name in hydra_cfg:
        return hydra_cfg[name]
    return default


def plot_premium_grid(
    data: np.ndarray,
    activation_prob: torch.Tensor,
    sort_idx: torch.Tensor,
    out_path: Path,
    *,
    cmap: str,
    title: str,
    filename: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    bg_color = "#1a1a1a"
    text_color = "#eeeeee"
    plt.rcParams.update(
        {
            "text.color": text_color,
            "axes.labelcolor": text_color,
            "xtick.color": text_color,
            "ytick.color": text_color,
        }
    )

    n_atoms = data.shape[0]
    n_plot = min(n_atoms, 64)
    cols = 8
    rows = (n_plot + cols - 1) // cols
    fig = plt.figure(figsize=(22, 2.5 * rows + 2), facecolor=bg_color)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 15], width_ratios=[40, 1])
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, title, fontsize=26, weight="bold", ha="center", va="center")

    border_cmap = mpl.colormaps["plasma"]
    border_norm = plt.Normalize(vmin=0.0, vmax=float(activation_prob.max().clamp_min(0.01)))

    for idx in range(n_plot):
        r = idx // cols
        c = idx % cols
        sub_ax = fig.add_axes([0.05 + c * 0.11, 0.05 + (rows - 1 - r) * 0.8 / rows, 0.10, 0.7 / rows])
        sub_ax.imshow(data[idx], aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        prob = float(activation_prob[idx])
        color = border_cmap(border_norm(prob))
        for spine in sub_ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        sub_ax.set_title(f"Atom {int(sort_idx[idx])} (p={prob:.3f})", fontsize=9, pad=2)

    ax_cbar = fig.add_subplot(gs[1, 1])
    sm = plt.cm.ScalarMappable(cmap=border_cmap, norm=border_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label(r"Activation Prob $\mathbb{E}[|\delta|]$", fontsize=16, labelpad=12)
    cbar.outline.set_edgecolor(text_color)

    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / filename, bbox_inches="tight", facecolor=bg_color, dpi=120)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Saved {out_path / filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ConvNMF spectrogram atoms")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-atoms", type=int, default=None)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--encoder-output-dim", type=int, default=None)
    parser.add_argument("--encoder-type", type=str, default=None)
    parser.add_argument("--decoder-type", type=str, default=None)
    parser.add_argument("--dict-init", type=str, default=None)
    parser.add_argument("--motif-width", type=int, default=None)
    parser.add_argument("--decoder-stride", type=int, default=None)
    parser.add_argument("--k-max", type=float, default=None)
    parser.add_argument("--k-min", type=float, default=None)
    parser.add_argument("--magnitude-dist", type=str, default=None)
    parser.add_argument("--structure-mode", type=str, default=None)
    parser.add_argument("--match-encoder-decoder-stride", action="store_true", default=None)
    parser.add_argument("--spectrogram-enhancements", dest="spectrogram_enhancements", action="store_true", default=None)
    parser.add_argument("--disable-spectrogram-enhancements", dest="spectrogram_enhancements", action="store_false")
    parser.add_argument("--denoise", action="store_true", default=None)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    hydra_cfg, cfg_path = load_hydra_config(ckpt_path)
    if cfg_path is not None:
        print(f"Using Hydra config defaults from {cfg_path}")
    state_dict = load_state_dict(ckpt_path, torch.device("cpu"))
    inferred = infer_arch_from_state_dict(state_dict)

    args.dataset = "fsdd"
    args.n_atoms = int(resolve_arg(args, hydra_cfg, inferred, "n_atoms", 128))
    args.input_channels = int(resolve_arg(args, hydra_cfg, inferred, "input_channels", 129))
    args.max_frames = int(resolve_arg(args, hydra_cfg, inferred, "max_frames", 64))
    args.input_length = int(resolve_arg(args, hydra_cfg, inferred, "input_length", args.max_frames))
    args.n_fft = int(resolve_arg(args, hydra_cfg, inferred, "n_fft", (args.input_channels - 1) * 2))
    args.hop_length = int(resolve_arg(args, hydra_cfg, inferred, "hop_length", 128))
    args.latent_dim = int(resolve_arg(args, hydra_cfg, inferred, "latent_dim", 64))
    args.encoder_output_dim = int(resolve_arg(args, hydra_cfg, inferred, "encoder_output_dim", 256))
    args.encoder_type = resolve_arg(args, hydra_cfg, inferred, "encoder_type", "resnet")
    args.decoder_type = resolve_arg(args, hydra_cfg, inferred, "decoder_type", "convnmf")
    args.dict_init = resolve_arg(args, hydra_cfg, inferred, "dict_init", "random")
    args.motif_width = int(resolve_arg(args, hydra_cfg, inferred, "motif_width", inferred.get("motif_width", 16)))
    args.decoder_stride = int(resolve_arg(args, hydra_cfg, inferred, "decoder_stride", 16))
    args.k_min = float(resolve_arg(args, hydra_cfg, inferred, "k_min", 0.1))
    args.k_max = float(resolve_arg(args, hydra_cfg, inferred, "k_max", 1e9))
    args.magnitude_dist = resolve_arg(args, hydra_cfg, inferred, "magnitude_dist", "gamma")
    args.structure_mode = resolve_arg(args, hydra_cfg, inferred, "structure_mode", "ternary")
    if args.match_encoder_decoder_stride is None:
        args.match_encoder_decoder_stride = bool(hydra_cfg.get("match_encoder_decoder_stride", False))
    if args.spectrogram_enhancements is None:
        args.spectrogram_enhancements = bool(hydra_cfg.get("spectrogram_enhancements", True))
    if args.denoise is None:
        args.denoise = bool(hydra_cfg.get("denoise", False))
    if args.decoder_type == "convnmf":
        args.input_length = args.max_frames

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    ds = get_fsdd_dataset(
        data_dir=str(REPO_ROOT / "data" / "fsdd"),
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_frames=args.max_frames,
        use_instance_norm=args.spectrogram_enhancements,
        denoise=args.denoise,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    batch_x, _ = next(iter(loader))
    batch_x = batch_x.to(device)

    with torch.no_grad():
        _, info = model(batch_x, temp=0.05, sampling="deterministic")
        delta = info["delta"]
        gamma = info.get("gamma", torch.ones_like(delta))
        activation_prob = delta.abs().mean(dim=(0, 2)).cpu()
        active_mask = delta.abs() > 0.5
        mag_when_active = (gamma.abs() * active_mask).sum(dim=(0, 2)) / (active_mask.sum(dim=(0, 2)) + 1e-6)
        mag_when_active = mag_when_active.cpu()

    with torch.no_grad():
        time_frames = args.input_length
        t_latent = max(1, time_frames // model.decoder.stride)
        baseline = model.decoder(torch.zeros(1, args.n_atoms, t_latent, device=device)).view(1, -1)
        scales = torch.where(mag_when_active > 0, mag_when_active, mag_when_active.mean().clamp_min(1.0))
        z_one_hot = torch.diag(scales).to(device)
        z_conv = torch.zeros(args.n_atoms, args.n_atoms, t_latent, device=device)
        z_conv[:, :, t_latent // 2] = z_one_hot
        atoms_recon = model.decoder(z_conv).view(args.n_atoms, -1)
        atoms_diff = atoms_recon - baseline

    sort_idx = torch.argsort(activation_prob, descending=True)
    activation_prob = activation_prob[sort_idx]
    atoms_recon = atoms_recon[sort_idx]
    atoms_diff = atoms_diff[sort_idx]

    freq_bins = args.input_channels
    actual_frames = atoms_recon.shape[-1] // freq_bins
    atoms_recon_2d = atoms_recon.view(args.n_atoms, freq_bins, actual_frames).cpu().numpy()
    atoms_diff_2d = atoms_diff.view(args.n_atoms, freq_bins, actual_frames).cpu().numpy()

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "figures_fsdd"
    plot_premium_grid(
        atoms_recon_2d,
        activation_prob,
        sort_idx,
        out_dir,
        cmap="magma",
        title="Spectrogram Atoms: full decoder motifs",
        filename="atoms_spectrograms.png",
    )
    max_abs = float(np.abs(atoms_diff_2d).max()) if atoms_diff_2d.size else 1.0
    plot_premium_grid(
        atoms_diff_2d,
        activation_prob,
        sort_idx,
        out_dir,
        cmap="bwr",
        title="Spectrogram strokes: atom minus decoder baseline",
        filename="atoms_strokes.png",
        vmin=-max_abs,
        vmax=max_abs,
    )


if __name__ == "__main__":
    main()
