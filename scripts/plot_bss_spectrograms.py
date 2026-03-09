#!/usr/bin/env python
"""Plot spectrogram diagnostics for BSS inference wav outputs.

Expected files in --inference-dir for a given utterance id:
  <utt>_mix.wav
  <utt>_mix_est.wav
  <utt>_s1_est.wav
  <utt>_s2_est.wav
  <utt>_s1_ref.wav
  <utt>_s2_ref.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, wav = wavfile.read(str(path))
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        scale = float(max(abs(info.min), info.max))
        wav = wav.astype(np.float32) / scale
    else:
        wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return sr, wav


def _stft_db(
    wav: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    db_floor: float,
) -> np.ndarray:
    x = torch.from_numpy(wav)
    win = torch.hann_window(win_length)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=win,
        center=True,
        return_complex=True,
    )
    mag = X.abs().clamp_min(1e-8)
    db = 20.0 * torch.log10(mag)
    db = torch.clamp(db, min=db_floor)
    return db.numpy()


def _find_utt_id(inference_dir: Path) -> str:
    mix_files = sorted(inference_dir.glob("*_mix.wav"))
    if not mix_files:
        raise FileNotFoundError(
            f"No *_mix.wav found in {inference_dir}. "
            "Run scripts/inference_bss.py first."
        )
    name = mix_files[0].stem
    if not name.endswith("_mix"):
        raise RuntimeError(f"Unexpected mix filename: {mix_files[0].name}")
    return name[:-4]


def _panel(
    ax: plt.Axes,
    spec_db: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    t_max = spec_db.shape[1] * hop_length / float(sr)
    f_max = sr / 2.0
    im = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0.0, t_max, 0.0, f_max],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (Hz)")
    return im


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BSS wav spectrogram diagnostics.")
    parser.add_argument("--inference-dir", type=str, required=True)
    parser.add_argument("--utt-id", type=str, default="", help="If omitted, auto-detect from *_mix.wav")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--db-floor", type=float, default=-80.0)
    parser.add_argument("--save-name", type=str, default="", help="Optional PNG filename")
    args = parser.parse_args()

    inf_dir = Path(args.inference_dir)
    if not inf_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inf_dir}")

    utt_id = args.utt_id if args.utt_id else _find_utt_id(inf_dir)

    paths = {
        "mix": inf_dir / f"{utt_id}_mix.wav",
        "mix_est": inf_dir / f"{utt_id}_mix_est.wav",
        "s1_est": inf_dir / f"{utt_id}_s1_est.wav",
        "s1_ref": inf_dir / f"{utt_id}_s1_ref.wav",
        "s2_est": inf_dir / f"{utt_id}_s2_est.wav",
        "s2_ref": inf_dir / f"{utt_id}_s2_ref.wav",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing expected wav files:\n" + "\n".join(missing))

    sr, mix = _read_wav(paths["mix"])
    sr2, mix_est = _read_wav(paths["mix_est"])
    sr3, s1_est = _read_wav(paths["s1_est"])
    sr4, s1_ref = _read_wav(paths["s1_ref"])
    sr5, s2_est = _read_wav(paths["s2_est"])
    sr6, s2_ref = _read_wav(paths["s2_ref"])

    if len({sr, sr2, sr3, sr4, sr5, sr6}) != 1:
        raise ValueError("Sample-rate mismatch across wav files.")

    specs = {
        "Mixture (input)": _stft_db(mix, args.n_fft, args.hop_length, args.win_length, args.db_floor),
        "Mixture reconstructed (s1_est+s2_est)": _stft_db(mix_est, args.n_fft, args.hop_length, args.win_length, args.db_floor),
        "Source 1 estimated": _stft_db(s1_est, args.n_fft, args.hop_length, args.win_length, args.db_floor),
        "Source 1 reference": _stft_db(s1_ref, args.n_fft, args.hop_length, args.win_length, args.db_floor),
        "Source 2 estimated": _stft_db(s2_est, args.n_fft, args.hop_length, args.win_length, args.db_floor),
        "Source 2 reference": _stft_db(s2_ref, args.n_fft, args.hop_length, args.win_length, args.db_floor),
    }

    all_vals = np.concatenate([s.reshape(-1) for s in specs.values()])
    vmin = max(args.db_floor, float(np.percentile(all_vals, 2)))
    vmax = float(np.percentile(all_vals, 99))

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    keys = list(specs.keys())
    ims = []
    for ax, key in zip(axes.flatten(), keys):
        im = _panel(
            ax=ax,
            spec_db=specs[key],
            sr=sr,
            hop_length=args.hop_length,
            title=key,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.95, pad=0.01)
    cbar.set_label("Magnitude (dB)")
    fig.suptitle(f"BSS Spectrogram Diagnostics - {utt_id}", fontsize=14)

    save_name = args.save_name if args.save_name else f"{utt_id}_spectrograms.png"
    out_path = inf_dir / save_name
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved spectrogram panel: {out_path}")


if __name__ == "__main__":
    main()

