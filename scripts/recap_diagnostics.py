#!/usr/bin/env python
"""Evaluate a trained model directory on Libri2Mix with SI-SDRi.

This script keeps the "folder-first" workflow:
  - reads checkpoint + optional train.log from a model directory
  - computes baseline run metrics (reconstruction/sparsity from log)
  - runs BSS evaluation on Libri2Mix and reports SI-SDR / SI-SDRi
  - writes a SOTA comparison table with placeholders
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.datasets import get_librimix_dataloader
from models.hybrid_vae import HybridSparseVAE
from scripts.inference_bss import separate_sources
from utils.separation import wiener_separation


def extract_final_train_metrics(train_log: Path) -> dict[str, float]:
    """Extract final epoch metrics from train.log if available.

    Supports the current logging format:
      k̄=... k_act=... n_act_frame=... n_act_total=.../N δ₀=...% β=... τ=... Δdict=...
    """
    if not train_log.exists():
        return {}

    def _extract(pattern: str, text: str) -> float | None:
        m = re.search(pattern, text)
        if not m:
            return None
        return float(m.group(1))

    with train_log.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    last_epoch_line = ""
    for line in reversed(lines):
        if "[INFO] - Epoch" in line:
            last_epoch_line = line
            break

    if not last_epoch_line:
        return {}

    metrics: dict[str, float] = {}

    epoch = _extract(r"Epoch\s+(\d+)", last_epoch_line)
    recon = _extract(r"recon\s+([0-9.]+)", last_epoch_line)
    kl_gamma = _extract(r"kl_γ\s+([0-9.]+)", last_epoch_line)
    kl_delta = _extract(r"kl_δ\s+([0-9.]+)", last_epoch_line)
    k_bar = _extract(r"k̄=([0-9.]+)", last_epoch_line)
    k_act = _extract(r"k_act=([0-9.]+)", last_epoch_line)
    n_act_frame = _extract(r"n_act_frame=([0-9.]+)", last_epoch_line)
    n_act_total = _extract(r"n_act_total=([0-9.]+)", last_epoch_line)
    delta_0_pct = _extract(r"δ₀=([0-9.]+)%", last_epoch_line)
    beta = _extract(r"β=([0-9.]+)", last_epoch_line)
    temp = _extract(r"τ=([0-9.]+)", last_epoch_line)
    dict_drift = _extract(r"Δdict=([0-9.]+)", last_epoch_line)

    if epoch is not None:
        metrics["epoch"] = epoch
    if recon is not None:
        metrics["recon"] = recon
    if kl_gamma is not None:
        metrics["kl_gamma"] = kl_gamma
    if kl_delta is not None:
        metrics["kl_delta"] = kl_delta
    if k_bar is not None:
        metrics["k_mean"] = k_bar
        metrics["k_bar_final"] = k_bar
    if k_act is not None:
        metrics["k_active"] = k_act
        metrics["k_act_final"] = k_act
    if n_act_frame is not None:
        metrics["n_active_frame"] = n_act_frame
        metrics["n_act_frame_final"] = n_act_frame
    if n_act_total is not None:
        metrics["n_active_total"] = n_act_total
        metrics["n_active"] = n_act_total  # Backward-compatible key
        metrics["n_act_total_final"] = n_act_total
    if delta_0_pct is not None:
        metrics["sparsity"] = delta_0_pct / 100.0
        metrics["delta_0_final"] = delta_0_pct
    if beta is not None:
        metrics["beta"] = beta
        metrics["beta_final"] = beta
    if temp is not None:
        metrics["temp"] = temp
    if dict_drift is not None:
        metrics["dict_drift"] = dict_drift

    return metrics


def load_hydra_config_from_model_dir(model_dir: Path) -> dict[str, Any]:
    """Load Hydra config from model_dir/.hydra/config.yaml when present."""
    cfg_path = model_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load plain state_dict or wrapped payload from checkpoint."""
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

    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def infer_arch_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Infer critical architecture dims directly from checkpoint tensors."""
    out: dict[str, int] = {}

    dec_w = state_dict.get("decoder.weight")
    if torch.is_tensor(dec_w) and dec_w.dim() == 3:
        # ConvNMF decoder weight: [n_atoms, freq_bins, motif_width]
        out["n_atoms"] = int(dec_w.shape[0])
        out["input_channels"] = int(dec_w.shape[1])
        out["motif_width"] = int(dec_w.shape[2])

    return out


def build_model_from_config(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> HybridSparseVAE:
    """Instantiate model without mutating architecture code."""
    inferred = infer_arch_from_state_dict(state_dict or {})

    input_channels = int(inferred.get("input_channels", cfg.get("input_channels", args.input_channels)))
    input_length_cfg = int(cfg.get("input_length", args.input_length))
    max_frames_cfg = cfg.get("max_frames")
    if isinstance(max_frames_cfg, int) and max_frames_cfg > 0:
        # Hydra config can keep stale input_length while actual training used max_frames.
        input_length = int(max_frames_cfg)
    else:
        input_length = input_length_cfg
    n_atoms = int(inferred.get("n_atoms", cfg.get("n_atoms", args.n_atoms)))
    motif_width = int(inferred.get("motif_width", cfg.get("motif_width", args.motif_width)))

    return HybridSparseVAE(
        input_channels=input_channels,
        input_length=input_length,
        encoder_type=cfg.get("encoder_type", args.encoder_type),
        encoder_output_dim=int(cfg.get("encoder_output_dim", args.encoder_output_dim)),
        n_atoms=n_atoms,
        latent_dim=int(cfg.get("latent_dim", args.latent_dim)),
        decoder_type=cfg.get("decoder_type", args.decoder_type),
        dict_init=cfg.get("dict_init", args.dict_init),
        normalize_dict=bool(cfg.get("normalize_dict", True)),
        k_min=float(cfg.get("k_min", args.k_min)),
        k_max=float(cfg.get("k_max", args.k_max)),
        magnitude_dist=cfg.get("magnitude_dist", args.magnitude_dist),
        structure_mode=cfg.get("structure_mode", args.structure_mode),
        motif_width=motif_width,
        decoder_stride=int(cfg.get("decoder_stride", args.decoder_stride)),
        match_encoder_decoder_stride=bool(
            cfg.get("match_encoder_decoder_stride", args.match_encoder_decoder_stride)
        ),
    )


def infer_two_source_magnitudes(
    model: HybridSparseVAE,
    mixture_mag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer two source magnitudes from explicit model outputs."""
    forward_out = model(mixture_mag, temp=0.05, sampling="deterministic")

    if isinstance(forward_out, tuple) and len(forward_out) == 2:
        x_recon, info = forward_out
    else:
        x_recon = forward_out
        info = {}

    if isinstance(info, dict):
        if "x_recon_s1" in info and "x_recon_s2" in info:
            return info["x_recon_s1"].clamp_min(0.0), info["x_recon_s2"].clamp_min(0.0)
        if "source_magnitudes" in info:
            src = info["source_magnitudes"]
            if src.dim() == 4 and src.shape[1] >= 2:
                return src[:, 0].clamp_min(0.0), src[:, 1].clamp_min(0.0)

    if torch.is_tensor(x_recon) and x_recon.dim() == 4 and x_recon.shape[1] >= 2:
        # Decoder already emits two source channels: [B,2,F,T]
        return x_recon[:, 0].clamp_min(0.0), x_recon[:, 1].clamp_min(0.0)

    raise RuntimeError(
        "Could not extract two source magnitudes from model output. "
        "Use --inference-method hybrid_affinity for unsupervised atom partitioning, "
        "or provide a model that emits (s1_mag, s2_mag)."
    )


def evaluate_librimix_bss(
    model: HybridSparseVAE,
    dataloader,
    device: torch.device,
    n_fft: int,
    hop_length: int,
    win_length: int,
    mask_power: float,
    max_eval: int,
    inference_method: str,
    alpha: float,
    h_representation: str,
) -> dict[str, float]:
    """Compute SI-SDR and SI-SDRi on Libri2Mix split."""
    model.eval()
    si_sdr_vals: list[float] = []
    si_sdri_vals: list[float] = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_eval > 0 and i >= max_eval:
                break

            mix_mag = batch["mixture_mag"].to(device)
            mix_c = batch["mixture_complex"].to(device)
            mix_wav = batch["mixture_wav"].to(device)
            s1_wav = batch["source1_wav"].to(device)
            s2_wav = batch["source2_wav"].to(device)
            lengths = batch["lengths"].to(device)

            if inference_method == "hybrid_affinity":
                for b in range(mix_mag.shape[0]):
                    l = int(lengths[b].item())
                    out = separate_sources(
                        model=model,
                        mix_mag=mix_mag[b : b + 1],
                        mix_complex=mix_c[b : b + 1],
                        alpha=alpha,
                        h_representation=h_representation,
                        mask_power=mask_power,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        length=l,
                    )
                    p1 = out["source1_waveform"][0, :l].to(device)
                    p2 = out["source2_waveform"][0, :l].to(device)
                    m = mix_wav[b, :l]
                    t1 = s1_wav[b, :l]
                    t2 = s2_wav[b, :l]

                    mix_ref = 0.5 * (
                        scale_invariant_signal_distortion_ratio(m, t1)
                        + scale_invariant_signal_distortion_ratio(m, t2)
                    )
                    perm_a = (
                        scale_invariant_signal_distortion_ratio(p1, t1)
                        + scale_invariant_signal_distortion_ratio(p2, t2)
                    ) * 0.5
                    perm_b = (
                        scale_invariant_signal_distortion_ratio(p1, t2)
                        + scale_invariant_signal_distortion_ratio(p2, t1)
                    ) * 0.5
                    best_sep = torch.maximum(perm_a, perm_b)

                    si_sdr_vals.append(float(best_sep.item()))
                    si_sdri_vals.append(float((best_sep - mix_ref).item()))
            else:
                est_s1_mag, est_s2_mag = infer_two_source_magnitudes(
                    model=model,
                    mixture_mag=mix_mag,
                )

                sep = wiener_separation(
                    mixture_complex=mix_c,
                    source1_mag=est_s1_mag,
                    source2_mag=est_s2_mag,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    length=lengths,
                    mask_power=mask_power,
                )
                e1 = sep["source1_waveform"].to(device)
                e2 = sep["source2_waveform"].to(device)

                for b in range(e1.shape[0]):
                    l = int(lengths[b].item())
                    m = mix_wav[b, :l]
                    t1 = s1_wav[b, :l]
                    t2 = s2_wav[b, :l]
                    p1 = e1[b, :l]
                    p2 = e2[b, :l]

                    mix_ref = 0.5 * (
                        scale_invariant_signal_distortion_ratio(m, t1)
                        + scale_invariant_signal_distortion_ratio(m, t2)
                    )
                    perm_a = (
                        scale_invariant_signal_distortion_ratio(p1, t1)
                        + scale_invariant_signal_distortion_ratio(p2, t2)
                    ) * 0.5
                    perm_b = (
                        scale_invariant_signal_distortion_ratio(p1, t2)
                        + scale_invariant_signal_distortion_ratio(p2, t1)
                    ) * 0.5
                    best_sep = torch.maximum(perm_a, perm_b)

                    si_sdr_vals.append(float(best_sep.item()))
                    si_sdri_vals.append(float((best_sep - mix_ref).item()))

    if not si_sdr_vals:
        raise RuntimeError("No Libri2Mix examples evaluated. Check dataset path/split.")

    return {
        "si_sdr_mean_db": float(sum(si_sdr_vals) / len(si_sdr_vals)),
        "si_sdri_mean_db": float(sum(si_sdri_vals) / len(si_sdri_vals)),
        "num_eval_examples": int(len(si_sdr_vals)),
    }


def write_sota_placeholder_table(out_csv: Path, ours_si_sdri: float) -> None:
    """Write NeurIPS-table scaffold with placeholders for external methods."""
    rows = [
        {"Model": "Hybrid Sparse VAE (Ours)", "Year": 2026, "Regime": "Unsupervised/Sparse", "SI-SDRi (dB)": f"{ours_si_sdri:.3f}"},
        {"Model": "SC-VAE", "Year": 2025, "Regime": "Unsupervised/Sparse", "SI-SDRi (dB)": "TODO"},
        {"Model": "DNMF-AG", "Year": 2025, "Regime": "Unsupervised/Sparse", "SI-SDRi (dB)": "TODO"},
        {"Model": "ArrayDPS", "Year": 2026, "Regime": "Unsupervised/Sparse", "SI-SDRi (dB)": "TODO"},
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Libri2Mix checkpoint-folder evaluation")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing hybrid_vae_final.pt")
    parser.add_argument("--checkpoint-name", type=str, default="hybrid_vae_final.pt")
    parser.add_argument("--librimix-root", type=str, default="./data/Libri2Mix")
    parser.add_argument("--librimix-split", type=str, default="test", choices=["train-100", "dev", "test"])
    parser.add_argument("--librimix-mix-type", type=str, default="min")
    parser.add_argument("--librimix-mixture-dirname", type=str, default="mix_clean")
    parser.add_argument("--librimix-sample-rate", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0, help="0 means full split")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--mask-power", type=float, default=2.0)
    parser.add_argument("--inference-method", type=str, default="hybrid_affinity", choices=["hybrid_affinity", "model_outputs"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--h-representation", type=str, default="B_abs", choices=["B_abs", "gamma"])

    # Model defaults if .hydra/config.yaml is absent
    parser.add_argument("--input-channels", type=int, default=257)
    parser.add_argument("--input-length", type=int, default=256)
    parser.add_argument("--encoder-type", type=str, default="resnet")
    parser.add_argument("--decoder-type", type=str, default="convnmf")
    parser.add_argument("--encoder-output-dim", type=int, default=256)
    parser.add_argument("--n-atoms", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dict-init", type=str, default="random")
    parser.add_argument("--k-min", type=float, default=0.1)
    parser.add_argument("--k-max", type=float, default=1e9)
    parser.add_argument("--magnitude-dist", type=str, default="gamma")
    parser.add_argument("--structure-mode", type=str, default="ternary")
    parser.add_argument("--motif-width", type=int, default=16)
    parser.add_argument("--decoder-stride", type=int, default=16)
    parser.add_argument("--match-encoder-decoder-stride", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    ckpt_path = model_dir / args.checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_hydra_config_from_model_dir(model_dir)
    state_dict = load_state_dict(ckpt_path, device=device)
    model = build_model_from_config(cfg, args, state_dict=state_dict).to(device)
    model.load_state_dict(state_dict)

    loader = get_librimix_dataloader(
        root_dir=args.librimix_root,
        split=args.librimix_split,
        sample_rate=args.librimix_sample_rate,
        mix_type=args.librimix_mix_type,
        mixture_dirname=args.librimix_mixture_dirname,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=None,  # SI-SDRi should run on full utterance length
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    librimix_metrics = evaluate_librimix_bss(
        model=model,
        dataloader=loader,
        device=device,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        mask_power=args.mask_power,
        max_eval=args.max_eval,
        inference_method=args.inference_method,
        alpha=args.alpha,
        h_representation=args.h_representation,
    )

    log_metrics = extract_final_train_metrics(model_dir / "train.log")
    summary = {
        "model_dir": str(model_dir),
        "checkpoint": str(ckpt_path),
        "inference": {
            "method": args.inference_method,
            "alpha": args.alpha,
            "h_representation": args.h_representation,
            "mask_power": args.mask_power,
        },
        "train_metrics_final": log_metrics,
        "libri2mix": {
            "split": args.librimix_split,
            "metrics": librimix_metrics,
        },
    }

    out_json = model_dir / "librimix_eval_metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_csv = model_dir / "librimix_sota_placeholder.csv"
    write_sota_placeholder_table(out_csv, ours_si_sdri=librimix_metrics["si_sdri_mean_db"])

    print("\n=== Libri2Mix Evaluation ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved metrics JSON: {out_json}")
    print(f"Saved SOTA placeholder table: {out_csv}")


if __name__ == "__main__":
    main()
