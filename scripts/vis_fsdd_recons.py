#!/usr/bin/env python
"""
FSDD Spectrogram Reconstruction & Audio Synthesis
1. Decodes a batch of FSDD samples.
2. Plots Original vs Reconstructed Spectrograms.
3. Synthesizes audio using Griffin-Lim from the magnitude spectrograms.
4. Saves .wav files for listening.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.io import wavfile

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize import build_model
from data.datasets import get_fsdd_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/fsdd_reconstructions")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Model/Data Params
    parser.add_argument("--input-length", type=int, default=8256)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--n-atoms", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--encoder-output-dim", type=int, default=256)
    parser.add_argument("--encoder-type", type=str, default="resnet")
    parser.add_argument("--decoder-type", type=str, default="convnmf")
    parser.add_argument("--dict-init", type=str, default="random")
    parser.add_argument("--magnitude-dist", type=str, default="gamma")
    parser.add_argument("--structure-mode", type=str, default="ternary")
    parser.add_argument("--disable-spectrogram-enhancements", action="store_false", dest="spectrogram_enhancements", help="Disable non-negativity and instance bounds.")
    parser.add_argument("--denoise", action="store_true", default=False, help="Apply Wiener-style spectral denoising.")
    parser.add_argument("--motif-width", type=int, default=16)
    parser.add_argument("--decoder-stride", type=int, default=16)
    parser.add_argument("--k-max", type=float, default=1e9)

    args, _ = parser.parse_known_args()
    args.dataset = 'fsdd'
    
    # Safety check for build_model requirements
    if not hasattr(args, 'encoder_output_dim'): args.encoder_output_dim = 256
    if not hasattr(args, 'dict_init'): args.dict_init = 'random'
    args.input_channels = args.n_fft // 2 + 1
    args.input_length = args.max_frames
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = build_model(args)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    # 2. Load Data
    print("Loading FSDD samples...")
    # Note: Dataset returns normalized magnitude spectrograms
    ds = get_fsdd_dataset(n_fft=args.n_fft, hop_length=args.hop_length, max_frames=args.max_frames,
                          use_instance_norm=args.spectrogram_enhancements,
                          denoise=args.denoise)
    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=True)
    batch_x, _ = next(iter(loader))
    batch_x = batch_x.to(device)

    # 3. Forward Pass
    print("Running reconstruction...")
    with torch.no_grad():
        recon_x, _ = model(batch_x, temp=0.05, sampling="deterministic")

    # 4. Reshape
    freq_bins = args.n_fft // 2 + 1
    orig_2d = batch_x.view(-1, freq_bins, args.max_frames).cpu().numpy()
    recon_2d = recon_x.view(-1, freq_bins, args.max_frames).cpu().numpy()

    # 5. Plot Comparison
    print("Plotting comparison...")
    fig, axes = plt.subplots(args.n_samples, 2, figsize=(10, 3 * args.n_samples))
    for i in range(args.n_samples):
        # Original
        axes[i, 0].imshow(orig_2d[i], aspect='auto', origin='lower', cmap='magma')
        axes[i, 0].set_title(f"Original {i}")
        axes[i, 0].axis('off')
        # Reconstructed
        axes[i, 1].imshow(recon_2d[i], aspect='auto', origin='lower', cmap='magma')
        axes[i, 1].set_title(f"Reconstructed {i}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path / "recon_comparison.png")
    print(f"✓ Saved comparison plot to {out_path / 'recon_comparison.png'}")

    # 6. Audio Synthesis (Griffin-Lim approximation)
    print("Synthesizing audio...")
    # Simplified Griffin-Lim: random phase + iterative refinement (or just random phase for quick check)
    # Since we only have magnitude, audio quality will be 'robotic' but recognizable.
    
    def griffin_lim(mag_norm, n_fft, hop_length, iterations=32):
        # Invert the log10([0, 1]) normalization applied in datasets.py
        # Assume mag_norm in [0, 1] represents an 80dB dynamic range (-80dB to 0dB)
        db_spec = mag_norm * 80.0 - 80.0
        mag_linear = 10.0 ** (db_spec / 20.0)
        
        # Start with random phase
        phase = np.exp(2j * np.pi * np.random.rand(*mag_linear.shape))
        complex_spec = mag_linear * phase
        
        # Iterative reconstruction (using CPU for simplicity)
        y = torch.istft(torch.from_numpy(complex_spec), n_fft=n_fft, hop_length=hop_length, 
                        window=torch.hann_window(n_fft), center=True)
        
        for _ in range(iterations):
            # STFT to get consistent phase
            stft = torch.stft(y, n_fft=n_fft, hop_length=hop_length, 
                              window=torch.hann_window(n_fft), center=True, return_complex=True)
            # Replace magnitude with the model's magnitude
            phase = torch.angle(stft)
            consistent_spec = torch.from_numpy(mag_linear) * torch.exp(1j * phase)
            # ISTFT back to time
            y = torch.istft(consistent_spec, n_fft=n_fft, hop_length=hop_length, 
                            window=torch.hann_window(n_fft), center=True)
        
        return y.numpy()

    fs = 8000 # FSDD default sampling rate
    for i in range(args.n_samples):
        # Rescale back from [0, 1] if needed (dataset normalizes)
        # Note: we don't have the original max, so we normalize the output for audio
        def save_audio(spec, name):
            audio = griffin_lim(spec, args.n_fft, args.hop_length)
            # Normalize for 16-bit PCM
            audio = audio / (np.max(np.abs(audio)) + 1e-9)
            wavfile.write(out_path / f"{name}.wav", fs, (audio * 32767).astype(np.int16))

        save_audio(orig_2d[i], f"sample_{i}_orig")
        save_audio(recon_2d[i], f"sample_{i}_recon")
        print(f"  - Saved audio for sample {i}")

    print(f"\nDone! Files available in {out_path}")

if __name__ == "__main__":
    main()
