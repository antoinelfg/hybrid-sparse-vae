#!/usr/bin/env python
"""Posthoc sparse-recovery evaluation for classical sinusoid baselines."""

from __future__ import annotations

import argparse
import json
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
    generate_sinusoid_recovery_batch,
    support_metrics_from_scores,
)
from scripts.baselines.run_baselines import build_dct_dictionary, omp


def compute_subspace_scores(atoms: torch.Tensor, max_frequency: int) -> tuple[torch.Tensor, torch.Tensor]:
    subspaces, _ = build_fourier_subspaces(atoms.shape[1], max_frequency=max_frequency)
    proj = torch.einsum("at,ftd->afd", atoms.float(), subspaces.float())
    scores = proj.norm(dim=-1).clamp(0.0, 1.0)
    best_score, _ = scores.max(dim=1)
    return scores, best_score


def evaluate_omp_condition(
    spec: SinusoidRecoverySpec,
    *,
    n_atoms: int,
    n_nonzero: int,
    min_atom_corr: float,
    min_subspace_score: float,
) -> dict[str, object]:
    batch = generate_sinusoid_recovery_batch(spec)
    x = batch["x"][:, 0].float()
    gt_support = batch["gt_support"]
    gt_amp_scores = batch["gt_amp_scores"]

    dictionary = build_dct_dictionary(n_atoms, spec.length).float()
    atoms = F.normalize(dictionary.T, p=2, dim=1)
    fourier_bank, labels = build_fourier_bank(spec.length, max_frequency=spec.max_frequency)
    corr = torch.abs(atoms @ fourier_bank.float().T)
    best_corr, best_idx = corr.max(dim=1)
    best_labels = [labels[int(idx)] for idx in best_idx]

    pred_scores = torch.zeros(spec.n_samples, spec.max_frequency, dtype=torch.float32)
    recon_sum = 0.0
    active_counts = []
    coeffs_all = []
    for sample_idx in range(spec.n_samples):
        coeffs = omp(x[sample_idx], dictionary, n_nonzero=n_nonzero).float()
        coeffs_all.append(coeffs)
        recon = dictionary @ coeffs
        recon_sum += float(F.mse_loss(recon, x[sample_idx], reduction="sum").item())
        active = coeffs != 0
        active_counts.append(float(active.float().sum().item()))
        for atom_idx in range(coeffs.numel()):
            if not bool(active[atom_idx]):
                continue
            if float(best_corr[atom_idx].item()) < min_atom_corr:
                continue
            _, freq_bin = best_labels[atom_idx]
            freq_idx = freq_bin - 1
            pred_scores[sample_idx, freq_idx] += float(coeffs[atom_idx].abs().item()) * float(best_corr[atom_idx].item())

    coeffs_tensor = torch.stack(coeffs_all, dim=0)
    support_eval = support_metrics_from_scores(pred_scores, gt_support, gt_amp_scores)

    subspace_scores, best_subspace_score = compute_subspace_scores(atoms, max_frequency=spec.max_frequency)
    pred_scores_subspace = torch.zeros_like(pred_scores)
    for atom_idx in range(coeffs_tensor.shape[1]):
        valid = subspace_scores[atom_idx] >= min_subspace_score
        if not bool(valid.any()):
            continue
        pred_scores_subspace += coeffs_tensor[:, atom_idx].abs().unsqueeze(1) * (
            subspace_scores[atom_idx] * valid.float()
        ).unsqueeze(0)
    subspace_eval = support_metrics_from_scores(pred_scores_subspace, gt_support, gt_amp_scores)

    return {
        "baseline_name": f"omp_{n_nonzero}",
        "dataset": "sinusoid",
        "dataset_spec": batch["spec"],
        "reconstruction": {
            "recon_mse_per_example": recon_sum / spec.n_samples,
            "recon_mse_mean": recon_sum / (spec.n_samples * spec.length),
        },
        "latents": {
            "collapsed": False,
            "n_active_frame_mean": sum(active_counts) / len(active_counts),
            "n_active_total_mean": sum(active_counts) / len(active_counts),
            "sparsity": float((coeffs_tensor == 0).float().mean().item()),
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
            **support_eval,
        },
        "subspace_eval": {
            "min_subspace_score": min_subspace_score,
            **subspace_eval,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate classical sparse baselines on sinusoid recovery")
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--selected-baseline", type=str, default="omp_33")
    parser.add_argument("--min-atom-corr", type=float, default=0.70)
    parser.add_argument("--min-subspace-score", type=float, default=0.70)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    spec = SinusoidRecoverySpec(
        n_samples=int(payload.get("n_samples", 512)),
        length=int(payload.get("length", 128)),
        n_components=int(payload.get("n_components", 3)),
        seed=int(payload.get("seed", 0)),
        max_frequency=19,
    )
    baselines = {
        "omp_33": evaluate_omp_condition(
            spec,
            n_atoms=128,
            n_nonzero=33,
            min_atom_corr=args.min_atom_corr,
            min_subspace_score=args.min_subspace_score,
        ),
        "omp_10": evaluate_omp_condition(
            spec,
            n_atoms=128,
            n_nonzero=10,
            min_atom_corr=args.min_atom_corr,
            min_subspace_score=args.min_subspace_score,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "seed": int(payload.get("seed", 0)),
                "selected_baseline": args.selected_baseline,
                "baselines": baselines,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
