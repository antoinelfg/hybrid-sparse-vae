import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_vae import HybridSparseVAE
from scripts.inference_bss import compute_hybrid_affinity, separate_sources


def test_compute_hybrid_affinity_shape_and_symmetry():
    n_atoms, f, w, t = 16, 64, 8, 20
    W = torch.rand(n_atoms, f, w)
    H = torch.rand(n_atoms, t)

    M = compute_hybrid_affinity(W=W, H=H, alpha=0.5)
    assert M.shape == (n_atoms, n_atoms)
    assert np.allclose(M, M.T, atol=1e-6)
    assert np.allclose(np.diag(M), 0.0, atol=1e-6)
    assert np.isfinite(M).all()


def test_separate_sources_smoke():
    model = HybridSparseVAE(
        input_channels=65,
        input_length=32,
        encoder_type="resnet",
        decoder_type="convnmf",
        n_atoms=16,
        latent_dim=8,
        motif_width=8,
        decoder_stride=4,
    ).eval()

    wav = torch.randn(1024)
    n_fft = 128
    hop = 32
    win = torch.hann_window(n_fft)
    mix_c = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    mix_mag = mix_c.abs().unsqueeze(0)  # [1,F,T]

    out = separate_sources(
        model=model,
        mix_mag=mix_mag,
        mix_complex=mix_c.unsqueeze(0),
        alpha=0.5,
        h_representation="B_abs",
        mask_power=2.0,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        length=wav.numel(),
    )

    assert out["source1_waveform"].shape[0] == 1
    assert out["source2_waveform"].shape[0] == 1
    assert out["source1_waveform"].shape[-1] >= wav.numel()
    assert out["source2_waveform"].shape[-1] >= wav.numel()
    assert len(np.unique(out["labels"])) == 2


def test_separate_sources_restores_decoder_output_length():
    model = HybridSparseVAE(
        input_channels=65,
        input_length=32,
        encoder_type="resnet",
        decoder_type="convnmf",
        n_atoms=16,
        latent_dim=8,
        motif_width=8,
        decoder_stride=4,
    ).eval()

    original_output_length = int(model.decoder.output_length)
    wav = torch.randn(2048)
    n_fft = 128
    hop = 32
    win = torch.hann_window(n_fft)
    mix_c = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    mix_mag = mix_c.abs().unsqueeze(0)

    separate_sources(
        model=model,
        mix_mag=mix_mag,
        mix_complex=mix_c.unsqueeze(0),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        length=wav.numel(),
    )

    assert int(model.decoder.output_length) == original_output_length
