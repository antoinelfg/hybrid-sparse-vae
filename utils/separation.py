"""Separation utilities for magnitude-domain source separation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_bft(x: torch.Tensor) -> torch.Tensor:
    """Normalize spectrogram tensor to [B, F, T]."""
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    if x.dim() == 4 and x.shape[1] == 1:
        return x.squeeze(1)
    raise ValueError(f"Expected [F,T], [B,F,T] or [B,1,F,T], got shape {tuple(x.shape)}")


def _align_time(spec: torch.Tensor, target_t: int) -> torch.Tensor:
    """Align [B, F, T] tensor to a target frame count with center crop/pad."""
    t = spec.shape[-1]
    if t == target_t:
        return spec
    if t > target_t:
        start = (t - target_t) // 2
        return spec[..., start:start + target_t]
    return F.pad(spec, (0, target_t - t))


def _stack_with_padding(wavs: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(w.shape[-1] for w in wavs)
    padded = []
    for wav in wavs:
        if wav.shape[-1] < max_len:
            wav = F.pad(wav, (0, max_len - wav.shape[-1]))
        padded.append(wav)
    return torch.stack(padded, dim=0)


def wiener_separation(
    mixture_complex: torch.Tensor,
    source1_mag: torch.Tensor,
    source2_mag: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int | None = None,
    length: int | torch.Tensor | list[int] | None = None,
    mask_power: float = 2.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Separate two sources by soft ratio/Wiener masking over mixture STFT.

    Parameters
    ----------
    mixture_complex:
        Complex STFT of mixture, shape [F,T] or [B,F,T].
    source1_mag, source2_mag:
        Predicted source magnitudes, same frequency bins as mixture.
    mask_power:
        1.0 -> soft mask, 2.0 -> Wiener-like power mask.

    Returns
    -------
    dict with:
      mask1, mask2               : [B,F,T]
      source1_complex, source2_complex : [B,F,T] complex
      source1_waveform, source2_waveform : [B,L]
    """
    mix_c = _to_bft(mixture_complex)
    s1 = _to_bft(source1_mag).clamp_min(0.0)
    s2 = _to_bft(source2_mag).clamp_min(0.0)

    if s1.shape[0] != mix_c.shape[0] or s2.shape[0] != mix_c.shape[0]:
        raise ValueError("Batch size mismatch between mixture and estimated magnitudes.")
    if s1.shape[1] != mix_c.shape[1] or s2.shape[1] != mix_c.shape[1]:
        raise ValueError("Frequency-bin mismatch between mixture and estimated magnitudes.")

    # Robust time alignment in case model output has slight frame mismatch.
    s1 = _align_time(s1, mix_c.shape[-1])
    s2 = _align_time(s2, mix_c.shape[-1])

    p1 = torch.pow(s1 + eps, mask_power)
    p2 = torch.pow(s2 + eps, mask_power)
    den = (p1 + p2).clamp_min(eps)

    mask1 = p1 / den
    mask2 = p2 / den

    est1_c = mask1.to(mix_c.dtype) * mix_c
    est2_c = mask2.to(mix_c.dtype) * mix_c

    win = torch.hann_window(win_length or n_fft, device=mix_c.device)

    if length is None:
        lengths = [None] * mix_c.shape[0]
    elif isinstance(length, int):
        lengths = [length] * mix_c.shape[0]
    elif isinstance(length, torch.Tensor):
        lengths = [int(v.item()) for v in length]
    else:
        lengths = [int(v) for v in length]

    s1_wavs: list[torch.Tensor] = []
    s2_wavs: list[torch.Tensor] = []
    for b in range(mix_c.shape[0]):
        s1_wav = torch.istft(
            est1_c[b],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            window=win,
            length=lengths[b],
        )
        s2_wav = torch.istft(
            est2_c[b],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            window=win,
            length=lengths[b],
        )
        s1_wavs.append(s1_wav)
        s2_wavs.append(s2_wav)

    return {
        "mask1": mask1,
        "mask2": mask2,
        "source1_complex": est1_c,
        "source2_complex": est2_c,
        "source1_waveform": _stack_with_padding(s1_wavs),
        "source2_waveform": _stack_with_padding(s2_wavs),
    }
