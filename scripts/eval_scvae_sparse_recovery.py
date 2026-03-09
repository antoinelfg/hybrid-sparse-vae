#!/usr/bin/env python
"""Evaluate a LISTA-style SC-VAE baseline on sinusoid sparse recovery."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sinusoid_recovery import (
    SinusoidRecoverySpec,
    build_fourier_bank,
    build_fourier_subspaces,
    collapse_flag,
    generate_sinusoid_recovery_batch,
    support_metrics_from_scores,
)
from scripts.baselines.run_scvae_sinusoid import LISTASCVAE


def load_checkpoint(path: Path) -> tuple[LISTASCVAE, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    cfg = dict(payload.get("config", {}))
    model = LISTASCVAE(
        input_dim=int(cfg.get("length", 128)),
        latent_dim=int(cfg.get("latent_dim", 128)),
        n_steps=int(cfg.get("n_steps", 5)),
        prior_scale=float(cfg.get("prior_scale", 0.25)),
        threshold_init=float(cfg.get("threshold_init", 0.1)),
    )
    state_dict = payload.get("model_state") or payload.get("state_dict") or payload
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def resolve_sinusoid_stress_arg(
    cli_value: object | None,
    checkpoint_cfg: dict[str, object],
    key: str,
    default: object,
) -> object:
    if cli_value is not None:
        return cli_value
    value = checkpoint_cfg.get(key)
    if value is not None:
        return value
    return default


def compute_subspace_scores(atoms: torch.Tensor, max_frequency: int) -> tuple[torch.Tensor, torch.Tensor]:
    subspaces, _ = build_fourier_subspaces(atoms.shape[1], max_frequency=max_frequency)
    proj = torch.einsum("at,ftd->afd", atoms.float(), subspaces.float())
    scores = proj.norm(dim=-1).clamp(0.0, 1.0)
    best_score, _ = scores.max(dim=1)
    return scores, best_score


def evaluate_scvae_sparse_recovery(
    checkpoint: Path,
    spec: SinusoidRecoverySpec,
    min_atom_corr: float,
    min_subspace_score: float,
) -> dict[str, object]:
    model, cfg = load_checkpoint(checkpoint)
    batch = generate_sinusoid_recovery_batch(spec)
    x = batch["x"]
    gt_support = batch["gt_support"]
    gt_amp_scores = batch["gt_amp_scores"]

    with torch.no_grad():
        recon, info = model(x)
    mu = info["mu"].detach().float()
    active = info["active"].detach()

    decoder_weight = model.decoder.net.weight.detach().float()
    atoms = F.normalize(decoder_weight.T, p=2, dim=1)
    bank, labels = build_fourier_bank(atoms.shape[1], max_frequency=spec.max_frequency)
    corr = torch.abs(atoms @ bank.float().T)
    best_corr, best_idx = corr.max(dim=1)
    best_labels = [labels[int(idx)] for idx in best_idx]

    pred_scores = torch.zeros(spec.n_samples, spec.max_frequency, dtype=torch.float32)
    for atom_idx in range(mu.shape[1]):
        if float(best_corr[atom_idx].item()) < min_atom_corr:
            continue
        _, freq_bin = best_labels[atom_idx]
        pred_scores[:, freq_bin - 1] += active[:, atom_idx].float() * mu[:, atom_idx].abs() * float(best_corr[atom_idx].item())

    support_metrics = support_metrics_from_scores(pred_scores, gt_support, gt_amp_scores)
    atom_subspace_scores, best_subspace_score = compute_subspace_scores(atoms, max_frequency=spec.max_frequency)
    pred_scores_subspace = torch.zeros_like(pred_scores)
    for atom_idx in range(mu.shape[1]):
        valid = atom_subspace_scores[atom_idx] >= min_subspace_score
        if not bool(valid.any()):
            continue
        pred_scores_subspace += mu[:, atom_idx].abs().unsqueeze(1) * active[:, atom_idx].float().unsqueeze(1) * (
            atom_subspace_scores[atom_idx] * valid.float()
        ).unsqueeze(0)
    subspace_metrics = support_metrics_from_scores(pred_scores_subspace, gt_support, gt_amp_scores)

    n_active = float(active.float().sum(dim=1).mean().item())
    recon_mse_per_example = float(F.mse_loss(recon, x, reduction="sum").item() / x.size(0))
    recon_mse_mean = float(F.mse_loss(recon, x, reduction="mean").item())

    return {
        "checkpoint": str(checkpoint),
        "dataset": "sinusoid",
        "dataset_spec": batch["spec"],
        "baseline_name": "scvae_lista",
        "config": cfg,
        "reconstruction": {
            "recon_mse_per_example": recon_mse_per_example,
            "recon_mse_mean": recon_mse_mean,
        },
        "latents": {
            "n_active_frame_mean": n_active,
            "n_active_total_mean": n_active,
            "k_mean": None,
            "k_active_mean": None,
            "sparsity": float((~active).float().mean().item()),
            "collapsed": collapse_flag(n_active),
            "threshold_mean": float(info["threshold_mean"]),
        },
        "atoms": {
            "n_atoms": int(atoms.shape[0]),
            "top1_fourier_corr_mean": float(best_corr.mean().item()),
            "top1_fourier_corr_median": float(best_corr.median().item()),
            "high_conf_atom_fraction": float((best_corr >= min_atom_corr).float().mean().item()),
            "best_subspace_score_mean": float(best_subspace_score.mean().item()),
            "best_subspace_score_median": float(best_subspace_score.median().item()),
        },
        "support_eval": {
            "min_atom_corr": min_atom_corr,
            **support_metrics,
        },
        "subspace_eval": {
            "min_subspace_score": min_subspace_score,
            **subspace_metrics,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LISTA-style SC-VAE sparse recovery")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frequency", type=int, default=19)
    parser.add_argument("--min-atom-corr", type=float, default=0.70)
    parser.add_argument("--min-subspace-score", type=float, default=0.70)
    parser.add_argument("--gain-distribution", type=str, default=None)
    parser.add_argument("--gain-min", type=float, default=None)
    parser.add_argument("--gain-max", type=float, default=None)
    parser.add_argument("--normalize-divisor", type=float, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _, checkpoint_cfg = load_checkpoint(args.checkpoint)
    spec = SinusoidRecoverySpec(
        n_samples=args.n_samples,
        length=args.length,
        n_components=args.n_components,
        seed=args.seed,
        max_frequency=args.max_frequency,
        gain_distribution=str(resolve_sinusoid_stress_arg(args.gain_distribution, checkpoint_cfg, "gain_distribution", "none")),
        gain_min=float(resolve_sinusoid_stress_arg(args.gain_min, checkpoint_cfg, "gain_min", 1.0)),
        gain_max=float(resolve_sinusoid_stress_arg(args.gain_max, checkpoint_cfg, "gain_max", 1.0)),
        normalize_divisor=float(
            resolve_sinusoid_stress_arg(args.normalize_divisor, checkpoint_cfg, "normalize_divisor", 4.0)
        ),
    )
    result = evaluate_scvae_sparse_recovery(
        checkpoint=args.checkpoint,
        spec=spec,
        min_atom_corr=args.min_atom_corr,
        min_subspace_score=args.min_subspace_score,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
