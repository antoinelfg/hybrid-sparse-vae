#!/usr/bin/env python
"""Render LibriMix experiment media from a checkpoint directory.

Outputs:
  - spectrogram_panel.png   : mixture / references / estimates
  - masks.png               : source masks or partition summaries when available
  - atoms_spectrograms.png  : shared decoder atoms (hybrid only)
  - atoms_strokes.png       : baseline-subtracted decoder atoms (hybrid only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import get_librimix_dataloader, get_librimix_dataset
from scripts.train_librimix_experiments import build_model, infer_input_shape, run_model
from utils.separation import wiener_separation


def _load_payload(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except Exception:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    return payload


def _resolve_checkpoint(path_like: Path) -> tuple[Path, Path]:
    if path_like.is_dir():
        for name in ("best.pt", "last.pt"):
            candidate = path_like / name
            if candidate.exists():
                return candidate, path_like
        raise FileNotFoundError(f"No best.pt/last.pt found in {path_like}")
    return path_like, path_like.parent


def _load_config(ckpt_dir: Path, payload: dict[str, Any]) -> argparse.Namespace:
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    elif "config" in payload and isinstance(payload["config"], dict):
        config = payload["config"]
    else:
        raise FileNotFoundError(f"Missing config.json and embedded config for {ckpt_dir}")
    return argparse.Namespace(**config)


def _load_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
    elif "model_state" in payload and isinstance(payload["model_state"], dict):
        state_dict = payload["model_state"]
    else:
        state_dict = payload
    if all(key.startswith("model.") for key in state_dict):
        state_dict = {key[len("model."):]: value for key, value in state_dict.items()}
    return state_dict


def _mag_to_db(spec: torch.Tensor, floor_db: float = -80.0) -> np.ndarray:
    spec = spec.detach().float().cpu().clamp_min(1e-8)
    db = 20.0 * torch.log10(spec)
    return torch.clamp(db, min=floor_db).numpy()


def _plot_spectrogram_panel(
    output_dir: Path,
    utt_id: str,
    mix_mag: torch.Tensor,
    mix_recon: torch.Tensor,
    s1_ref: torch.Tensor,
    s1_est: torch.Tensor,
    s2_ref: torch.Tensor,
    s2_est: torch.Tensor,
) -> None:
    specs = {
        "Mixture": _mag_to_db(mix_mag),
        "Mixture Recon": _mag_to_db(mix_recon),
        "Source 1 Ref": _mag_to_db(s1_ref),
        "Source 1 Est": _mag_to_db(s1_est),
        "Source 2 Ref": _mag_to_db(s2_ref),
        "Source 2 Est": _mag_to_db(s2_est),
    }
    all_vals = np.concatenate([value.reshape(-1) for value in specs.values()])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 99))

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for ax, (title, value) in zip(axes.flatten(), specs.items()):
        ax.imshow(value, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"LibriMix spectrogram diagnostics: {utt_id}", fontsize=14)
    fig.savefig(output_dir / "spectrogram_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_masks(
    output_dir: Path,
    utt_id: str,
    masks: torch.Tensor | None,
    source_assign: torch.Tensor | None,
) -> None:
    panels: list[tuple[str, np.ndarray]] = []
    if masks is not None:
        panels.extend(
            [
                ("Mask 1", masks[0].detach().cpu().numpy()),
                ("Mask 2", masks[1].detach().cpu().numpy()),
            ]
        )
    if source_assign is not None:
        panels.extend(
            [
                ("Assign 1 (mean over time)", source_assign[0].detach().cpu().numpy()),
                ("Assign 2 (mean over time)", source_assign[1].detach().cpu().numpy()),
            ]
        )
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, value) in zip(axes, panels):
        ax.imshow(value, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"LibriMix masks/assignments: {utt_id}", fontsize=13)
    fig.savefig(output_dir / "masks.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_atom_grid(
    output_dir: Path,
    atoms: np.ndarray,
    activation_prob: torch.Tensor,
    filename: str,
    title: str,
    cmap: str,
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

    n_atoms = atoms.shape[0]
    n_plot = min(n_atoms, 64)
    cols = 8
    rows = (n_plot + cols - 1) // cols
    fig = plt.figure(figsize=(22, 2.6 * rows + 2), facecolor=bg_color)
    border_cmap = mpl.colormaps["plasma"]
    border_norm = plt.Normalize(vmin=0.0, vmax=float(activation_prob.max().clamp_min(0.01)))

    fig.suptitle(title, fontsize=24, fontweight="bold", color=text_color)
    for idx in range(n_plot):
        r = idx // cols
        c = idx % cols
        ax = fig.add_axes([0.05 + c * 0.11, 0.08 + (rows - 1 - r) * 0.82 / rows, 0.10, 0.72 / rows])
        ax.imshow(atoms[idx], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        color = border_cmap(border_norm(float(activation_prob[idx])))
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_title(f"p={float(activation_prob[idx]):.3f}", fontsize=8, pad=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=140, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def _render_hybrid_atoms(
    model: torch.nn.Module,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    input_length: int,
) -> None:
    loader = get_librimix_dataloader(
        root_dir=args.librimix_root,
        split=args.val_split,
        sample_rate=args.sample_rate,
        mix_type=args.mix_type,
        mixture_dirname=args.mixture_dirname,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=args.max_frames,
        crop_mode=args.crop_mode,
        batch_size=min(4, max(1, int(getattr(args, "batch_size", 4)))),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    mix_mag = batch["mixture_mag"].to(device)
    with torch.no_grad():
        out, _ = run_model(model, mix_mag, args, eval_mode=True)
        active_mask = (out["source_acts"].sum(dim=1) > 0).float()
        activation_prob = active_mask.mean(dim=(0, 2)).cpu()
        act_sum = out["source_acts"].sum(dim=1)
        avg_mag = (act_sum.sum(dim=(0, 2)) / (active_mask.sum(dim=(0, 2)) + 1e-6)).cpu()

        decoder = model.backbone.decoder
        t_latent = max(1, int(np.ceil(input_length / max(decoder.stride, 1))))
        baseline = decoder(torch.zeros(1, args.n_atoms, t_latent, device=device))
        scales = torch.where(avg_mag > 0, avg_mag, avg_mag.mean().clamp_min(1.0))
        z_one_hot = torch.diag(scales).to(device)
        z_conv = torch.zeros(args.n_atoms, args.n_atoms, t_latent, device=device)
        z_conv[:, :, t_latent // 2] = z_one_hot
        atoms_recon = decoder(z_conv)
        atoms_diff = atoms_recon - baseline

    sort_idx = torch.argsort(activation_prob, descending=True)
    activation_prob = activation_prob[sort_idx]
    atoms_recon = atoms_recon[sort_idx]
    atoms_diff = atoms_diff[sort_idx]
    atoms_recon_np = atoms_recon.detach().cpu().numpy()
    atoms_diff_np = atoms_diff.detach().cpu().numpy()
    max_abs = float(np.abs(atoms_diff_np).max()) if atoms_diff_np.size else 1.0

    _plot_atom_grid(
        output_dir=output_dir,
        atoms=atoms_recon_np,
        activation_prob=activation_prob,
        filename="atoms_spectrograms.png",
        title="LibriMix hybrid atoms: shared decoder motifs",
        cmap="magma",
    )
    _plot_atom_grid(
        output_dir=output_dir,
        atoms=atoms_diff_np,
        activation_prob=activation_prob,
        filename="atoms_strokes.png",
        title="LibriMix hybrid atom strokes: atom minus decoder baseline",
        cmap="bwr",
        vmin=-max_abs,
        vmax=max_abs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LibriMix experiment media")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file or checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--index", type=int, default=0, help="Dataset item index from val split")
    args_cli = parser.parse_args()

    ckpt_path, ckpt_dir = _resolve_checkpoint(Path(args_cli.checkpoint))
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else ckpt_dir / "figures_librimix"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    payload = _load_payload(ckpt_path, device)
    args = _load_config(ckpt_dir, payload)
    state_dict = _load_state_dict(payload)

    train_dataset = get_librimix_dataset(
        root_dir=args.librimix_root,
        split=args.train_split,
        sample_rate=args.sample_rate,
        mix_type=args.mix_type,
        mixture_dirname=args.mixture_dirname,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=args.max_frames,
        crop_mode=args.crop_mode,
    )
    input_channels, input_length = infer_input_shape(train_dataset)
    model = build_model(args, input_channels=input_channels, input_length=input_length)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    eval_dataset = get_librimix_dataset(
        root_dir=args.librimix_root,
        split=args.val_split,
        sample_rate=args.sample_rate,
        mix_type=args.mix_type,
        mixture_dirname=args.mixture_dirname,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=None,
        crop_mode="center",
    )
    sample = eval_dataset[args_cli.index]
    mix_mag = sample["mixture_mag"].unsqueeze(0).to(device)
    mix_c = sample["mixture_complex"].unsqueeze(0).to(device)
    s1_mag = sample["source1_mag"]
    s2_mag = sample["source2_mag"]

    with torch.no_grad():
        out, _ = run_model(model, mix_mag, args, eval_mode=True)
        est_sources = out["source_mags"]
        mix_recon = out.get("mixture_recon", est_sources.sum(dim=1))
        sep = wiener_separation(
            mixture_complex=mix_c,
            source1_mag=est_sources[:, 0],
            source2_mag=est_sources[:, 1],
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            length=sample["length"],
            mask_power=args.mask_power,
        )

    _plot_spectrogram_panel(
        output_dir=output_dir,
        utt_id=str(sample["utt_id"]),
        mix_mag=sample["mixture_mag"],
        mix_recon=mix_recon[0].cpu(),
        s1_ref=s1_mag,
        s1_est=est_sources[0, 0].cpu(),
        s2_ref=s2_mag,
        s2_est=est_sources[0, 1].cpu(),
    )

    masks = out.get("masks")
    source_assign = out.get("source_assign")
    if masks is not None:
        masks = masks[0]
    if source_assign is not None:
        source_assign = source_assign[0].mean(dim=-1, keepdim=True)
    _plot_masks(output_dir, str(sample["utt_id"]), masks, source_assign)

    if args.experiment == "hybrid_partition":
        _render_hybrid_atoms(model, args, output_dir, device, input_length=input_length)

    metadata = {
        "utt_id": sample["utt_id"],
        "checkpoint": str(ckpt_path),
        "experiment": args.experiment,
        "output_dir": str(output_dir),
        "waveform_length": int(sample["length"]),
        "separated_waveforms_shape": {
            "source1": list(sep["source1_waveform"].shape),
            "source2": list(sep["source2_waveform"].shape),
        },
    }
    with (output_dir / "render_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"Saved LibriMix media to {output_dir}")


if __name__ == "__main__":
    main()
