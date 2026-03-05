import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.separation import wiener_separation


def test_wiener_separation_improves_over_mixture():
    sr = 8000
    n_fft = 256
    hop = 64
    t = torch.linspace(0, 1.0, sr)

    s1 = 0.5 * torch.sin(2 * torch.pi * 220.0 * t)
    s2 = 0.4 * torch.sin(2 * torch.pi * 440.0 * t)
    mix = s1 + s2

    win = torch.hann_window(n_fft)
    mix_c = torch.stft(mix, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    s1_mag = torch.stft(s1, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs()
    s2_mag = torch.stft(s2, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs()

    out = wiener_separation(
        mixture_complex=mix_c,
        source1_mag=s1_mag,
        source2_mag=s2_mag,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        length=s1.numel(),
        mask_power=2.0,
    )

    e1 = out["source1_waveform"][0]
    e2 = out["source2_waveform"][0]

    # Account for permutation ambiguity.
    err_a = torch.mean((e1 - s1) ** 2) + torch.mean((e2 - s2) ** 2)
    err_b = torch.mean((e1 - s2) ** 2) + torch.mean((e2 - s1) ** 2)
    sep_err = torch.minimum(err_a, err_b)

    mix_err = torch.mean((mix - s1) ** 2) + torch.mean((mix - s2) ** 2)
    assert sep_err < mix_err
