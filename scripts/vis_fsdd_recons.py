#!/usr/bin/env python
"""Visualize FSDD spectrogram reconstructions from a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FSDD reconstructions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--n-atoms", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--encoder-output-dim", type=int, default=None)
    parser.add_argument("--encoder-type", type=str, default=None)
    parser.add_argument("--decoder-type", type=str, default=None)
    parser.add_argument("--dict-init", type=str, default=None)
    parser.add_argument("--magnitude-dist", type=str, default=None)
    parser.add_argument("--structure-mode", type=str, default=None)
    parser.add_argument("--motif-width", type=int, default=None)
    parser.add_argument("--decoder-stride", type=int, default=None)
    parser.add_argument("--k-min", type=float, default=None)
    parser.add_argument("--k-max", type=float, default=None)
    parser.add_argument("--match-encoder-decoder-stride", action="store_true", default=None)
    parser.add_argument("--spectrogram-enhancements", dest="spectrogram_enhancements", action="store_true", default=None)
    parser.add_argument("--disable-spectrogram-enhancements", dest="spectrogram_enhancements", action="store_false")
    parser.add_argument("--denoise", action="store_true", default=None)

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
    args.magnitude_dist = resolve_arg(args, hydra_cfg, inferred, "magnitude_dist", "gamma")
    args.structure_mode = resolve_arg(args, hydra_cfg, inferred, "structure_mode", "ternary")
    args.motif_width = int(resolve_arg(args, hydra_cfg, inferred, "motif_width", 16))
    args.decoder_stride = int(resolve_arg(args, hydra_cfg, inferred, "decoder_stride", 16))
    args.k_min = float(resolve_arg(args, hydra_cfg, inferred, "k_min", 0.1))
    args.k_max = float(resolve_arg(args, hydra_cfg, inferred, "k_max", 1e9))
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

    dataset = get_fsdd_dataset(
        data_dir=str(REPO_ROOT / "data" / "fsdd"),
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_frames=args.max_frames,
        use_instance_norm=args.spectrogram_enhancements,
        denoise=args.denoise,
    )
    loader = DataLoader(dataset, batch_size=args.n_samples, shuffle=True)
    batch_x, _ = next(iter(loader))
    batch_x = batch_x.to(device)

    with torch.no_grad():
        recon_x, _ = model(batch_x, temp=0.05, sampling="deterministic")

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "figures_fsdd_recons"
    out_dir.mkdir(parents=True, exist_ok=True)

    orig = batch_x.cpu().numpy()
    recon = recon_x.cpu().numpy()
    vmax = float(max(orig.max(), recon.max()))
    vmin = float(min(orig.min(), recon.min()))

    fig, axes = plt.subplots(args.n_samples, 2, figsize=(10, 2.8 * args.n_samples), constrained_layout=True)
    if args.n_samples == 1:
        axes = np.array([axes])  # type: ignore[name-defined]
    for idx in range(args.n_samples):
        axes[idx, 0].imshow(orig[idx], aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        axes[idx, 0].set_title(f"Original {idx}")
        axes[idx, 1].imshow(recon[idx], aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        axes[idx, 1].set_title(f"Recon {idx}")
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_yticks([])
        axes[idx, 1].set_xticks([])
        axes[idx, 1].set_yticks([])
    fig.suptitle("FSDD spectrogram reconstructions", fontsize=14)
    fig.savefig(out_dir / "recon_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'recon_comparison.png'}")


if __name__ == "__main__":
    main()
