#!/usr/bin/env python
"""Unsupervised BSS inference via hybrid atom affinity + spectral clustering.

Core idea:
  1) infer sparse atom activations on the mixture,
  2) build a hybrid affinity matrix between atoms:
        spectral timbre similarity + temporal co-activation similarity,
  3) cluster atoms into 2 groups,
  4) reconstruct 2 source magnitudes by zeroing complementary atom groups,
  5) apply soft/Wiener masking on mixture complex STFT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.io import wavfile
from sklearn.cluster import KMeans, SpectralClustering

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.librimix_dataset import LibriMixDataset
from models.hybrid_vae import HybridSparseVAE
from utils.separation import wiener_separation


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except Exception:
        payload = torch.load(path, map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")

    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def load_hydra_config(ckpt_path: Path) -> dict[str, Any]:
    search_roots = [ckpt_path.parent, *ckpt_path.parents]
    for root in search_roots:
        cfg = root / ".hydra" / "config.yaml"
        if cfg.exists():
            with cfg.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        if root == REPO_ROOT:
            break
    return {}


def build_model_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> HybridSparseVAE:
    return HybridSparseVAE(
        input_channels=int(cfg.get("input_channels", args.input_channels)),
        input_length=int(cfg.get("input_length", args.input_length)),
        encoder_type=cfg.get("encoder_type", args.encoder_type),
        encoder_output_dim=int(cfg.get("encoder_output_dim", args.encoder_output_dim)),
        n_atoms=int(cfg.get("n_atoms", args.n_atoms)),
        latent_dim=int(cfg.get("latent_dim", args.latent_dim)),
        decoder_type=cfg.get("decoder_type", args.decoder_type),
        dict_init=cfg.get("dict_init", args.dict_init),
        normalize_dict=bool(cfg.get("normalize_dict", True)),
        k_min=float(cfg.get("k_min", args.k_min)),
        k_max=float(cfg.get("k_max", args.k_max)),
        magnitude_dist=cfg.get("magnitude_dist", args.magnitude_dist),
        structure_mode=cfg.get("structure_mode", args.structure_mode),
        motif_width=int(cfg.get("motif_width", args.motif_width)),
        decoder_stride=int(cfg.get("decoder_stride", args.decoder_stride)),
    )


def _infer_latent_info(model: HybridSparseVAE, mix_mag: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return latent info dict for a mixture magnitude [B,F,T]."""
    if model.lista_mode:
        k, theta, logits = model.encoder(mix_mag)
        _, info = model.latent.forward_from_params(k, theta, logits, temp=0.05, sampling="deterministic")
    else:
        h = model.encoder(mix_mag)
        _, info = model.latent(h, temp=0.05, sampling="deterministic")
    return info


def compute_hybrid_affinity(
    W: torch.Tensor,
    H: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Hybrid affinity matrix over atoms.

    Parameters
    ----------
    W:
        Decoder atoms [n_atoms, freq_bins, motif_width].
    H:
        Atom activations [n_atoms, time_frames].
    alpha:
        Fusion weight between spectral and temporal affinities.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    W_profile = W.abs().mean(dim=-1)  # [n_atoms, freq_bins]
    W_norm = F.normalize(W_profile, p=2, dim=-1, eps=eps)
    M_spec = torch.mm(W_norm, W_norm.t()).clamp_min(0.0)

    H_norm = F.normalize(H, p=2, dim=-1, eps=eps)
    M_temp = torch.mm(H_norm, H_norm.t()).clamp_min(0.0)

    M = alpha * M_spec + (1.0 - alpha) * M_temp
    M = 0.5 * (M + M.t())
    M.fill_diagonal_(0.0)
    return M.detach().cpu().numpy()


def _safe_cluster_atoms(affinity: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Cluster atoms into 2 groups from an affinity matrix."""
    n_atoms = affinity.shape[0]
    if n_atoms < 2:
        raise ValueError("Need at least 2 atoms for two-source clustering.")

    if not np.isfinite(affinity).all() or np.allclose(affinity, 0.0):
        # Fallback: deterministic split.
        labels = np.zeros(n_atoms, dtype=np.int64)
        labels[n_atoms // 2 :] = 1
        return labels

    try:
        clustering = SpectralClustering(
            n_clusters=2,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        )
        labels = clustering.fit_predict(affinity)
    except Exception:
        # Robust fallback using rows as features.
        km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
        labels = km.fit_predict(affinity)

    # Guard against degenerate 1-cluster output.
    if len(np.unique(labels)) < 2:
        labels = np.zeros(n_atoms, dtype=np.int64)
        labels[n_atoms // 2 :] = 1
    return labels.astype(np.int64)


def separate_sources(
    model: HybridSparseVAE,
    mix_mag: torch.Tensor,
    mix_complex: torch.Tensor,
    alpha: float = 0.5,
    h_representation: str = "B_abs",
    mask_power: float = 2.0,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int | None = None,
    length: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run full BSS inference for one mixture."""
    if not model.temporal_mode:
        raise ValueError("This inference pipeline expects decoder_type='convnmf'.")

    if mix_mag.dim() == 2:
        mix_mag = mix_mag.unsqueeze(0)
    if mix_complex.dim() == 2:
        mix_complex = mix_complex.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        info = _infer_latent_info(model, mix_mag)
        B = info["B"]  # [1, n_atoms, T']
        gamma = info["gamma"]  # [1, n_atoms, T']

        if h_representation == "gamma":
            H = gamma.squeeze(0).clamp_min(0.0)
        elif h_representation == "B_abs":
            H = B.squeeze(0).abs()
        else:
            raise ValueError(f"Unsupported h_representation: {h_representation}")

        if not hasattr(model.decoder, "weight"):
            raise ValueError("Model decoder has no conv atoms weight.")
        W = model.decoder.weight.detach()  # [n_atoms, freq_bins, motif_width]

        affinity = compute_hybrid_affinity(W=W, H=H, alpha=alpha)
        labels = _safe_cluster_atoms(affinity=affinity, random_state=random_state)

        omega_1 = np.where(labels == 0)[0]
        omega_2 = np.where(labels == 1)[0]

        B_1 = B.clone()
        B_1[:, omega_2, :] = 0.0
        B_2 = B.clone()
        B_2[:, omega_1, :] = 0.0

        V_1 = model.decoder(B_1).clamp_min(0.0)  # [1,F,T]
        V_2 = model.decoder(B_2).clamp_min(0.0)

        sep = wiener_separation(
            mixture_complex=mix_complex,
            source1_mag=V_1,
            source2_mag=V_2,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            length=length,
            mask_power=mask_power,
        )

    return {
        "labels": labels,
        "affinity": affinity,
        "omega_1": omega_1,
        "omega_2": omega_2,
        "source1_complex": sep["source1_complex"],
        "source2_complex": sep["source2_complex"],
        "source1_waveform": sep["source1_waveform"],
        "source2_waveform": sep["source2_waveform"],
        "source1_mag": V_1,
        "source2_mag": V_2,
    }


def _write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    wav = audio.detach().cpu().float().numpy()
    mx = float(np.max(np.abs(wav)))
    if mx > 0:
        wav = wav / mx
    wav_i16 = (wav * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, wav_i16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-mixture BSS inference with hybrid affinity.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--librimix-root", type=str, default="./data/Libri2Mix")
    parser.add_argument("--split", type=str, default="test", choices=["train-100", "dev", "test"])
    parser.add_argument("--mix-type", type=str, default="min")
    parser.add_argument("--mixture-dirname", type=str, default="mix_clean")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mask-power", type=float, default=2.0)
    parser.add_argument("--h-representation", type=str, default="B_abs", choices=["B_abs", "gamma"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/inference_bss")
    parser.add_argument("--save-affinity", action="store_true")

    # Fallback model config if no hydra config exists near checkpoint.
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
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_hydra_config(ckpt)
    model = build_model_from_config(cfg, args).to(device)
    model.load_state_dict(load_state_dict(ckpt, device))

    ds = LibriMixDataset(
        root_dir=args.librimix_root,
        split=args.split,
        sample_rate=args.sample_rate,
        mix_type=args.mix_type,
        mixture_dirname=args.mixture_dirname,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=None,
    )
    if not (0 <= args.sample_index < len(ds)):
        raise IndexError(f"sample-index={args.sample_index} out of range [0,{len(ds)-1}]")

    sample = ds[args.sample_index]
    mix_mag = sample["mixture_mag"].unsqueeze(0).to(device)
    mix_c = sample["mixture_complex"].unsqueeze(0).to(device)
    length = sample["length"]

    out = separate_sources(
        model=model,
        mix_mag=mix_mag,
        mix_complex=mix_c,
        alpha=args.alpha,
        h_representation=args.h_representation,
        mask_power=args.mask_power,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        length=length,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    utt = sample["utt_id"]
    sr = sample["sample_rate"]

    _write_wav(out_dir / f"{utt}_mix.wav", sample["mixture_wav"], sr)
    _write_wav(out_dir / f"{utt}_s1_est.wav", out["source1_waveform"][0, :length], sr)
    _write_wav(out_dir / f"{utt}_s2_est.wav", out["source2_waveform"][0, :length], sr)
    _write_wav(out_dir / f"{utt}_s1_ref.wav", sample["source1_wav"], sr)
    _write_wav(out_dir / f"{utt}_s2_ref.wav", sample["source2_wav"], sr)

    summary = {
        "utt_id": utt,
        "alpha": args.alpha,
        "mask_power": args.mask_power,
        "h_representation": args.h_representation,
        "n_atoms_cluster0": int((out["labels"] == 0).sum()),
        "n_atoms_cluster1": int((out["labels"] == 1).sum()),
        "files": {
            "mix": str(out_dir / f"{utt}_mix.wav"),
            "s1_est": str(out_dir / f"{utt}_s1_est.wav"),
            "s2_est": str(out_dir / f"{utt}_s2_est.wav"),
            "s1_ref": str(out_dir / f"{utt}_s1_ref.wav"),
            "s2_ref": str(out_dir / f"{utt}_s2_ref.wav"),
        },
    }
    with (out_dir / f"{utt}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.save_affinity:
        np.save(out_dir / f"{utt}_affinity.npy", out["affinity"])
        np.save(out_dir / f"{utt}_labels.npy", out["labels"])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
