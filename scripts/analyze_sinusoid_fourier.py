#!/usr/bin/env python
"""Analyze whether a strict-linear sinusoid checkpoint learned Fourier-like atoms.

This script is intentionally conservative:
  - it only treats checkpoints whose final decoder is strictly linear,
  - it computes the effective atom basis in signal space,
  - it matches each atom against a sine/cosine Fourier bank,
  - it optionally evaluates deterministic reconstruction on the toy dataset.

It supports the older sinusoid checkpoints stored as raw state_dicts with:
  * encoder.net.*
  * latent.fc_params.*
  * latent.dictionary.weight
  * decoder.net.0.*
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import generate_toy_sinusoid_tensors


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    if all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def is_strict_linear_decoder(state_dict: dict[str, torch.Tensor]) -> bool:
    decoder_keys = sorted(k for k in state_dict if k.startswith("decoder."))
    return decoder_keys == ["decoder.net.0.bias", "decoder.net.0.weight"]


def is_old_mlp_decoder(state_dict: dict[str, torch.Tensor]) -> bool:
    decoder_keys = sorted(k for k in state_dict if k.startswith("decoder."))
    return decoder_keys == [
        "decoder.net.0.bias",
        "decoder.net.0.weight",
        "decoder.net.2.bias",
        "decoder.net.2.weight",
        "decoder.net.4.bias",
        "decoder.net.4.weight",
    ]


def normalized_dictionary(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    weight = state_dict["latent.dictionary.weight"].detach().float()
    return F.normalize(weight, p=2, dim=0)


def effective_atoms_signal_space(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    dict_w = normalized_dictionary(state_dict)  # [latent_dim, n_atoms]
    dec_w = state_dict["decoder.net.0.weight"].detach().float()  # [T, latent_dim]
    eff = dec_w @ dict_w  # [T, n_atoms]
    return F.normalize(eff.T, p=2, dim=1)  # [n_atoms, T]


def decoder_forward_old_nonlinear(state_dict: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    w0, b0 = state_dict["decoder.net.0.weight"], state_dict["decoder.net.0.bias"]
    w2, b2 = state_dict["decoder.net.2.weight"], state_dict["decoder.net.2.bias"]
    w4, b4 = state_dict["decoder.net.4.weight"], state_dict["decoder.net.4.bias"]
    h = F.linear(z, w0.float(), b0.float())
    h = F.relu(h)
    h = F.linear(h, w2.float(), b2.float())
    h = F.relu(h)
    return F.linear(h, w4.float(), b4.float())


def nonlinear_atom_responses_signal_space(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    dict_w = normalized_dictionary(state_dict)  # [latent_dim, n_atoms]
    z0 = torch.zeros(1, dict_w.shape[0], dtype=torch.float32)
    baseline = decoder_forward_old_nonlinear(state_dict, z0).squeeze(0)  # [T]

    atoms = []
    for j in range(dict_w.shape[1]):
        z = dict_w[:, j].unsqueeze(0)  # [1, latent_dim]
        curve = decoder_forward_old_nonlinear(state_dict, z).squeeze(0) - baseline
        atoms.append(curve)
    atoms = torch.stack(atoms, dim=0)
    return F.normalize(atoms, p=2, dim=1)


def build_fourier_bank(length: int) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    n = torch.arange(length, dtype=torch.float32)
    bank: list[torch.Tensor] = []
    labels: list[tuple[str, int]] = []
    for k in range(1, length // 2 + 1):
        s = torch.sin(2 * math.pi * k * n / length)
        c = torch.cos(2 * math.pi * k * n / length)
        bank.append(F.normalize(s, p=2, dim=0))
        labels.append(("sin", k))
        bank.append(F.normalize(c, p=2, dim=0))
        labels.append(("cos", k))
    return torch.stack(bank, dim=0), labels


def build_fourier_subspaces(length: int) -> tuple[torch.Tensor, list[int]]:
    """Return orthonormal sin/cos subspaces per discrete frequency.

    Output shape: [n_freq, T, 2]
    """
    n = torch.arange(length, dtype=torch.float32)
    subspaces = []
    freqs = []
    for k in range(1, min(20, length // 2 + 1)):
        s = torch.sin(2 * math.pi * k * n / length)
        c = torch.cos(2 * math.pi * k * n / length)
        basis = torch.stack([s, c], dim=1)  # [T, 2]
        q, _ = torch.linalg.qr(basis, mode="reduced")
        subspaces.append(q)
        freqs.append(k)
    return torch.stack(subspaces, dim=0), freqs


def manual_deterministic_recon(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    k_min: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Old linear encoder: Linear -> ReLU -> Linear -> ReLU -> Linear
    w0, b0 = state_dict["encoder.net.0.weight"], state_dict["encoder.net.0.bias"]
    w2, b2 = state_dict["encoder.net.2.weight"], state_dict["encoder.net.2.bias"]
    w4, b4 = state_dict["encoder.net.4.weight"], state_dict["encoder.net.4.bias"]
    fc_w, fc_b = state_dict["latent.fc_params.weight"], state_dict["latent.fc_params.bias"]
    dict_w = normalized_dictionary(state_dict)
    x_flat = x.view(x.size(0), -1).float()
    h = F.linear(x_flat, w0.float(), b0.float())
    h = F.relu(h)
    h = F.linear(h, w2.float(), b2.float())
    h = F.relu(h)
    h = F.linear(h, w4.float(), b4.float())

    params = F.linear(h, fc_w.float(), fc_b.float())
    n_atoms = dict_w.shape[1]
    raw_k = params[:, :n_atoms]
    raw_theta = params[:, n_atoms : 2 * n_atoms]
    logits = params[:, 2 * n_atoms :].view(x.size(0), n_atoms, 3)

    k = F.softplus(raw_k) + k_min
    theta = F.softplus(raw_theta) + 1e-6
    gamma = k * theta
    idx = logits.argmax(dim=-1)
    delta = F.one_hot(idx, 3).float()[..., 2] - F.one_hot(idx, 3).float()[..., 0]
    b = gamma * delta
    z = F.linear(b, dict_w.float())
    if is_strict_linear_decoder(state_dict):
        dec_w = state_dict["decoder.net.0.weight"].detach().float()
        dec_b = state_dict["decoder.net.0.bias"].detach().float()
        recon = F.linear(z, dec_w, dec_b).view(x.size(0), 1, -1)
    elif is_old_mlp_decoder(state_dict):
        recon = decoder_forward_old_nonlinear(state_dict, z).view(x.size(0), 1, -1)
    else:
        raise RuntimeError("Unsupported decoder structure in old-style checkpoint.")
    return recon, {"k": k, "delta": delta, "gamma": gamma, "B": b}


def manual_deterministic_recon_current(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    k_min: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Current linear encoder: same 3-layer MLP
    w0, b0 = state_dict["encoder.net.0.weight"], state_dict["encoder.net.0.bias"]
    w2, b2 = state_dict["encoder.net.2.weight"], state_dict["encoder.net.2.bias"]
    w4, b4 = state_dict["encoder.net.4.weight"], state_dict["encoder.net.4.bias"]
    cp_w, cp_b = state_dict["latent.conv_params.weight"], state_dict["latent.conv_params.bias"]
    dict_w = normalized_dictionary(state_dict)
    dec_w = state_dict["decoder.net.0.weight"].detach().float()
    dec_b = state_dict["decoder.net.0.bias"].detach().float()

    x_flat = x.view(x.size(0), -1).float()
    h = F.linear(x_flat, w0.float(), b0.float())
    h = F.relu(h)
    h = F.linear(h, w2.float(), b2.float())
    h = F.relu(h)
    h = F.linear(h, w4.float(), b4.float())

    # conv_params is a 1x1 Conv1d on [B, C, 1]
    params = F.conv1d(h.unsqueeze(-1), cp_w.float(), cp_b.float()).squeeze(-1)
    n_atoms = dict_w.shape[1]
    raw_k = params[:, :n_atoms]
    raw_theta = params[:, n_atoms : 2 * n_atoms]
    logits = params[:, 2 * n_atoms :].view(x.size(0), n_atoms, 3)

    k = F.softplus(raw_k) + k_min
    theta = F.softplus(raw_theta) + 1e-6
    gamma = k * theta
    idx = logits.argmax(dim=-1)
    delta = F.one_hot(idx, 3).float()[..., 2] - F.one_hot(idx, 3).float()[..., 0]
    b = gamma * delta
    z = F.linear(b, dict_w.float())
    recon = F.linear(z, dec_w, dec_b).view(x.size(0), 1, -1)
    return recon, {"k": k, "delta": delta, "gamma": gamma, "B": b}


def plot_atoms(
    atoms: torch.Tensor,
    best_labels: list[tuple[str, int]],
    best_corr: torch.Tensor,
    output_path: Path,
) -> None:
    n_atoms, _ = atoms.shape
    n_cols = 4
    n_rows = math.ceil(n_atoms / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.6 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(n_atoms):
        ax = axes[i]
        ax.plot(atoms[i].numpy(), lw=1.75)
        basis, freq = best_labels[i]
        ax.set_title(f"a{i} | {basis}{freq} | r={best_corr[i]:.3f}", fontsize=9)
        ax.grid(alpha=0.2)

    for j in range(n_atoms, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Effective signal-space atoms matched to Fourier basis", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((x @ y / denom).item())


def compute_frequency_recovery_metrics(
    freqs: torch.Tensor,
    amps: torch.Tensor,
    info: dict[str, torch.Tensor],
    best_labels: list[tuple[str, int]],
    best_corr: torch.Tensor,
    min_atom_corr: float,
) -> dict[str, float]:
    """Evaluate recovery of true frequency support using atom-to-frequency matches."""
    b = info["B"].detach().float()
    active = (info["delta"].detach() != 0)
    n_samples, n_atoms = b.shape
    max_freq = 19

    pred_scores = torch.zeros(n_samples, max_freq, dtype=torch.float32)
    gt_amp_scores = torch.zeros(n_samples, max_freq, dtype=torch.float32)
    gt_support = torch.zeros(n_samples, max_freq, dtype=torch.bool)

    for s in range(n_samples):
        for comp_idx in range(freqs.shape[1]):
            f_idx = int(freqs[s, comp_idx].item()) - 1
            gt_support[s, f_idx] = True
            gt_amp_scores[s, f_idx] += float(amps[s, comp_idx].item())

    for atom_idx in range(n_atoms):
        freq_kind, freq_bin = best_labels[atom_idx]
        _ = freq_kind  # retained for future phase-aware evaluation
        if float(best_corr[atom_idx].item()) < min_atom_corr:
            continue
        f_idx = int(freq_bin) - 1
        pred_scores[:, f_idx] += active[:, atom_idx].float() * b[:, atom_idx].abs() * float(best_corr[atom_idx].item())

    pred_support_hard = pred_scores > 0.0

    precision_vals = []
    recall_vals = []
    f1_vals = []
    topk_precision_vals = []
    topk_recall_vals = []
    topk_f1_vals = []
    amp_corr_vals = []
    pred_support_sizes = []
    gt_support_sizes = []

    for s in range(n_samples):
        gt = gt_support[s]
        pred = pred_support_hard[s]
        tp = int((gt & pred).sum().item())
        pp = int(pred.sum().item())
        gp = int(gt.sum().item())

        precision = tp / pp if pp > 0 else 0.0
        recall = tp / gp if gp > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
        pred_support_sizes.append(float(pp))
        gt_support_sizes.append(float(gp))

        k = max(gp, 1)
        topk_idx = torch.topk(pred_scores[s], k=k).indices
        pred_topk = torch.zeros_like(gt)
        pred_topk[topk_idx] = True
        tp_topk = int((gt & pred_topk).sum().item())
        precision_topk = tp_topk / k
        recall_topk = tp_topk / gp if gp > 0 else 0.0
        f1_topk = (
            2 * precision_topk * recall_topk / (precision_topk + recall_topk)
            if (precision_topk + recall_topk) > 0
            else 0.0
        )
        topk_precision_vals.append(precision_topk)
        topk_recall_vals.append(recall_topk)
        topk_f1_vals.append(f1_topk)

        amp_corr_vals.append(_pearson_corr(pred_scores[s], gt_amp_scores[s]))

    return {
        "freq_eval_min_atom_corr": float(min_atom_corr),
        "gt_support_size_mean": float(torch.tensor(gt_support_sizes).mean().item()),
        "pred_support_size_mean": float(torch.tensor(pred_support_sizes).mean().item()),
        "support_precision_mean": float(torch.tensor(precision_vals).mean().item()),
        "support_recall_mean": float(torch.tensor(recall_vals).mean().item()),
        "support_f1_mean": float(torch.tensor(f1_vals).mean().item()),
        "topk_precision_mean": float(torch.tensor(topk_precision_vals).mean().item()),
        "topk_recall_mean": float(torch.tensor(topk_recall_vals).mean().item()),
        "topk_f1_mean": float(torch.tensor(topk_f1_vals).mean().item()),
        "pred_gt_amplitude_corr_mean": float(torch.tensor(amp_corr_vals).mean().item()),
    }


def compute_subspace_recovery_metrics(
    atoms: torch.Tensor,
    freqs: torch.Tensor,
    amps: torch.Tensor,
    info: dict[str, torch.Tensor],
    min_subspace_score: float,
) -> dict[str, float]:
    """Evaluate frequency recovery using sin/cos subspaces instead of single atoms.

    This metric is phase-invariant: an atom only needs to lie in the 2D subspace
    spanned by sin(k t) and cos(k t).
    """
    subspaces, subspace_freqs = build_fourier_subspaces(atoms.shape[1])  # [F, T, 2]
    # Projection norm of each atom on each frequency subspace: [n_atoms, n_freq]
    proj = torch.einsum("at,ftd->afd", atoms.float(), subspaces.float())
    atom_subspace_scores = proj.norm(dim=-1).clamp(0.0, 1.0)
    best_subspace_score, best_subspace_idx = atom_subspace_scores.max(dim=1)

    b = info["B"].detach().float()
    active = (info["delta"].detach() != 0)
    n_samples, n_atoms = b.shape
    n_freq = len(subspace_freqs)

    pred_scores = torch.zeros(n_samples, n_freq, dtype=torch.float32)
    gt_amp_scores = torch.zeros(n_samples, n_freq, dtype=torch.float32)
    gt_support = torch.zeros(n_samples, n_freq, dtype=torch.bool)

    for s in range(n_samples):
        for comp_idx in range(freqs.shape[1]):
            f_raw = int(freqs[s, comp_idx].item())
            if f_raw in subspace_freqs:
                f_idx = subspace_freqs.index(f_raw)
                gt_support[s, f_idx] = True
                gt_amp_scores[s, f_idx] += float(amps[s, comp_idx].item())

    for atom_idx in range(n_atoms):
        valid = atom_subspace_scores[atom_idx] >= min_subspace_score
        if not bool(valid.any()):
            continue
        contrib = active[:, atom_idx].float() * b[:, atom_idx].abs()
        pred_scores += contrib.unsqueeze(1) * (atom_subspace_scores[atom_idx] * valid.float()).unsqueeze(0)

    pred_support_hard = pred_scores > 0.0
    precision_vals = []
    recall_vals = []
    f1_vals = []
    topk_precision_vals = []
    topk_recall_vals = []
    topk_f1_vals = []
    amp_corr_vals = []
    pred_support_sizes = []
    gt_support_sizes = []

    for s in range(n_samples):
        gt = gt_support[s]
        pred = pred_support_hard[s]
        tp = int((gt & pred).sum().item())
        pp = int(pred.sum().item())
        gp = int(gt.sum().item())

        precision = tp / pp if pp > 0 else 0.0
        recall = tp / gp if gp > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
        pred_support_sizes.append(float(pp))
        gt_support_sizes.append(float(gp))

        k = max(gp, 1)
        topk_idx = torch.topk(pred_scores[s], k=k).indices
        pred_topk = torch.zeros_like(gt)
        pred_topk[topk_idx] = True
        tp_topk = int((gt & pred_topk).sum().item())
        precision_topk = tp_topk / k
        recall_topk = tp_topk / gp if gp > 0 else 0.0
        f1_topk = (
            2 * precision_topk * recall_topk / (precision_topk + recall_topk)
            if (precision_topk + recall_topk) > 0
            else 0.0
        )
        topk_precision_vals.append(precision_topk)
        topk_recall_vals.append(recall_topk)
        topk_f1_vals.append(f1_topk)
        amp_corr_vals.append(_pearson_corr(pred_scores[s], gt_amp_scores[s]))

    return {
        "subspace_min_score": float(min_subspace_score),
        "best_subspace_score_mean": float(best_subspace_score.mean().item()),
        "best_subspace_score_median": float(best_subspace_score.median().item()),
        "n_atoms_subspace_ge_0_95": int((best_subspace_score >= 0.95).sum().item()),
        "n_atoms_subspace_ge_0_90": int((best_subspace_score >= 0.90).sum().item()),
        "subspace_pred_support_size_mean": float(torch.tensor(pred_support_sizes).mean().item()),
        "subspace_gt_support_size_mean": float(torch.tensor(gt_support_sizes).mean().item()),
        "subspace_support_precision_mean": float(torch.tensor(precision_vals).mean().item()),
        "subspace_support_recall_mean": float(torch.tensor(recall_vals).mean().item()),
        "subspace_support_f1_mean": float(torch.tensor(f1_vals).mean().item()),
        "subspace_topk_precision_mean": float(torch.tensor(topk_precision_vals).mean().item()),
        "subspace_topk_recall_mean": float(torch.tensor(topk_recall_vals).mean().item()),
        "subspace_topk_f1_mean": float(torch.tensor(topk_f1_vals).mean().item()),
        "subspace_pred_gt_amplitude_corr_mean": float(torch.tensor(amp_corr_vals).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Fourier-like structure in sinusoid checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k-min", type=float, default=0.01)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--min-atom-corr", type=float, default=0.70)
    parser.add_argument("--min-subspace-score", type=float, default=0.70)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    state_dict = load_state_dict(ckpt_path)
    if is_strict_linear_decoder(state_dict):
        atoms = effective_atoms_signal_space(state_dict)  # [n_atoms, T]
        analysis_mode = "strict_linear_effective_atoms"
    elif is_old_mlp_decoder(state_dict):
        atoms = nonlinear_atom_responses_signal_space(state_dict)
        analysis_mode = "nonlinear_single_atom_responses"
    else:
        raise RuntimeError(
            f"Unsupported decoder structure for checkpoint {ckpt_path}."
        )

    bank, labels = build_fourier_bank(atoms.shape[1])
    corr = torch.abs(atoms @ bank.T)
    best_corr, best_idx = corr.max(dim=1)
    best_labels = [labels[int(i)] for i in best_idx]

    X, freqs, amps, phases = generate_toy_sinusoid_tensors(
        n_samples=args.n_samples,
        length=atoms.shape[1],
        n_components=3,
        seed=args.seed,
    )
    if "latent.fc_params.weight" in state_dict:
        recon, info = manual_deterministic_recon(state_dict, X, k_min=args.k_min)
    elif "latent.conv_params.weight" in state_dict:
        recon, info = manual_deterministic_recon_current(state_dict, X, k_min=args.k_min)
    else:
        raise RuntimeError("Unsupported latent parameterization for sinusoid Fourier analysis.")

    mse_mean_per_pixel = ((recon - X) ** 2).mean().item()
    mse_sum_per_signal = ((recon - X) ** 2).sum(dim=(1, 2)).mean().item()
    delta0 = float((info["delta"] == 0).float().mean().item())
    n_active = float((info["delta"] != 0).float().sum(dim=1).mean().item())
    mutual = torch.abs(atoms @ atoms.T)
    mutual.fill_diagonal_(0.0)
    freq_recovery = compute_frequency_recovery_metrics(
        freqs=freqs,
        amps=amps,
        info=info,
        best_labels=best_labels,
        best_corr=best_corr,
        min_atom_corr=args.min_atom_corr,
    )
    subspace_recovery = compute_subspace_recovery_metrics(
        atoms=atoms,
        freqs=freqs,
        amps=amps,
        info=info,
        min_subspace_score=args.min_subspace_score,
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "analysis_mode": analysis_mode,
        "n_atoms": int(atoms.shape[0]),
        "signal_length": int(atoms.shape[1]),
        "mse_mean_per_pixel": mse_mean_per_pixel,
        "mse_sum_per_signal": mse_sum_per_signal,
        "delta0": delta0,
        "n_active_mean": n_active,
        "k_mean": float(info["k"].mean().item()),
        "best_fourier_corr_mean": float(best_corr.mean().item()),
        "best_fourier_corr_median": float(best_corr.median().item()),
        "best_fourier_corr_min": float(best_corr.min().item()),
        "n_atoms_corr_ge_0_95": int((best_corr >= 0.95).sum().item()),
        "n_atoms_corr_ge_0_90": int((best_corr >= 0.90).sum().item()),
        "atom_mutual_coherence_mean": float(mutual.mean().item()),
        "atom_mutual_coherence_max": float(mutual.max().item()),
        **freq_recovery,
        **subspace_recovery,
        "top_matches": [
            {
                "atom": int(i),
                "corr": float(best_corr[i].item()),
                "basis": best_labels[i][0],
                "freq_bin": int(best_labels[i][1]),
            }
            for i in torch.argsort(best_corr, descending=True)[: min(16, atoms.shape[0])]
        ],
    }

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "fourier_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fourier_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_atoms(atoms, best_labels, best_corr, output_dir / "effective_atoms_vs_fourier.png")

    print(json.dumps(summary, indent=2))
    print(f"Saved summary -> {json_path}")
    print(f"Saved figure  -> {output_dir / 'effective_atoms_vs_fourier.png'}")


if __name__ == "__main__":
    main()
