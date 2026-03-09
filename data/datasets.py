"""Dataset loaders for real-data experiments (MNIST and Audio).

This module provides PyTorch datasets that return `TensorDataset` or
similar structures compatible with the training loop in `train.py`.
"""

import torch
from torch.utils.data import TensorDataset
from pathlib import Path


def get_mnist_dataset(data_dir: str = "./data", flatten: bool = True) -> TensorDataset:
    """Load MNIST as a 1D signal dataset.

    Parameters
    ----------
    data_dir : str
        Directory to download/store the MNIST dataset.
    flatten : bool
        If True, each 28x28 image is flattened into a 1x784 signal.
        If False, returns 1x28x28 images.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise ImportError(
            "torchvision is required for MNIST dataset. Please install it with `pip install torchvision`."
        ) from e

    # Simple transform to tensor and normalize to roughly [-1, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    
    # Pre-extract all data to a TensorDataset to fit the current training loop design
    # which assumes the dataset's `.tensors[0]` is accessible, or at least supports
    # returning a batch of shape `[B, C, T]`.
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, _ = next(iter(loader))  # X is [60000, 1, 28, 28]

    if flatten:
        X = X.view(X.shape[0], 1, -1)  # [60000, 1, 784]

    return TensorDataset(X)


def get_audio_spectrogram_dataset(
    data_dir: str = "./data/audio", 
    n_samples: int = 1000,
    n_mels: int = 64,
    time_steps: int = 128,
    use_instance_norm: bool = True
) -> TensorDataset:
    """Generate or load an audio spectrogram dataset.
    
    Currently this generates synthetic 'speech-like' spectrograms using
    Perlin noise/harmonic structures since downloading a real dataset
    (like SpeechCommands or LibriSpeech) requires external dependencies
    and disk space. Will be updated to use Torchaudio in the future.
    """
    # Create simple synthetic spectrograms for testing the pipeline
    # Shape should be [N, C, T] -> [N, n_mels, time_steps]
    # For a 1D signal model over time, we might treat it as [N, n_mels, time_steps]
    # where n_mels is the "channels" or we flatten it.
    
    # We will generate smooth pseudo-spectrogram random data
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_mels, time_steps)
    
    # Smooth over time to look more like audio
    kernel = torch.ones(1, 1, 5) / 5.0
    # Apply padding to maintain size
    import torch.nn.functional as F
    X = F.conv1d(X.view(-1, 1, time_steps), kernel, padding=2).view(n_samples, n_mels, time_steps)
    
    if use_instance_norm:
        # Normalize magnitudes to [0, 1] range to work well with linear decoders
        X = X - X.amin(dim=(1, 2), keepdim=True)
        X_max = X.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        X = X / X_max
    else:
        # Global normalization to [-1, 1]
        X = (X - X.min()) / (X.max() - X.min()) * 2.0 - 1.0

    return TensorDataset(X)


def get_fsdd_dataset(
    data_dir: str = "./data/fsdd",
    n_fft: int = 256,
    hop_length: int = 128,
    max_frames: int = 64,
    use_instance_norm: bool = True,
    denoise: bool = False,
    denoise_factor: float = 2.0,
) -> TensorDataset:
    """Download FSDD and return flattened magnitude spectrograms.
    
    The Free Spoken Digit Dataset audio files are processed using STFT.
    The resulting spectrograms are padded/cropped to a fixed `max_frames` 
    duration and flattened to 1D vectors for a linear generic autoencoder.

    Parameters
    ----------
    data_dir : str
        Directory to store the FSDD repository clone.
    n_fft : int
        Size of FFT to use for STFT. Returns (n_fft // 2 + 1) frequency bins.
    hop_length : int
        Number of audio samples between adjacent STFT columns.
    max_frames : int
        Target number of time frames for the spectrogram. Shorter audio
        is zero-padded; longer audio is cropped.

    Returns
    -------
    TensorDataset
        Dataset containing exactly one tensor of shape
        `[N, 1, Freq * Time]`, where `Freq = n_fft // 2 + 1` and `Time = max_frames`.
    """
    import os
    import subprocess
    import torch.nn.functional as F
    import scipy.io.wavfile as wavfile

    repo_url = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
    fsdd_dir = Path(data_dir)
    
    if not fsdd_dir.exists():
        fsdd_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading FSDD to {fsdd_dir}...")
        subprocess.run(["git", "clone", repo_url, str(fsdd_dir)], check=True)
    
    recordings_dir = fsdd_dir / "recordings"
    if not recordings_dir.exists() or not recordings_dir.is_dir():
        raise RuntimeError(f"FSDD recordings directory not found at {recordings_dir}")
        
    wav_files = list(recordings_dir.glob("*.wav"))
    if len(wav_files) == 0:
        raise RuntimeError(f"No .wav files found in {recordings_dir}")
        
    spectrograms = []
    labels = []
    freq_bins = n_fft // 2 + 1
    
    # Process each audio file
    for wav_file in sorted(wav_files):
        # Extract label from filename (e.g. 0_jackson_0.wav -> 0)
        label = int(wav_file.name.split('_')[0])
        labels.append(label)
        
        # Load audio using scipy
        sample_rate, data = wavfile.read(str(wav_file))
        # Ensure it's a float32 tensor of shape [1, length]
        waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        
        # Optionally normalize waveform if data was integer type
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()
        
        # Compute STFT
        window = torch.hann_window(n_fft)
        stft = torch.stft(
            waveform, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window,
            return_complex=True
        )
        
        # Take magnitude [1, Freq, Time]
        mag = torch.abs(stft)
        
        # --- Trim Silence ---
        energy = mag.sum(dim=1).squeeze(0)  # [Time]
        threshold = energy.max() * 0.05 # 5% of max energy
        active_frames = torch.where(energy > threshold)[0]
        if len(active_frames) > 0:
            start_idx = active_frames[0].item()
            end_idx = active_frames[-1].item()
            mag = mag[..., start_idx:end_idx+1]

        # --- Optional Spectral Denoising (Wiener-style) ---
        # Estimate stationary noise floor per frequency bin as the median magnitude
        # across time. This is robust (unlike mean) to transient peaks.
        # Soft-threshold: any bin below denoise_factor * noise_floor is zeroed.
        if denoise:
            noise_floor = mag.median(dim=-1, keepdim=True).values  # [1, Freq, 1]
            threshold_val = denoise_factor * noise_floor
            mag = torch.relu(mag - threshold_val)  # soft-threshold (NMF-friendly, keeps >=0)
            
        # Apply Logarithmic compression (Log-Spectrogram) to manage acoustic dynamic range
        # and prevent NMF decoders from failing on extreme linear peaks.
        mag = torch.log10(mag + 1e-6)
        
        # We need uniform length across time dimension
        time_len = mag.shape[-1]
        
        if time_len < max_frames:
            # Center pad with minimum value instead of 0s because of log scale
            pad_amount = max_frames - time_len
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            min_val = mag.min()
            mag = F.pad(mag, (pad_left, pad_right), value=min_val.item())
        elif time_len > max_frames:
            # Center crop to max_frames
            crop_amount = time_len - max_frames
            crop_left = crop_amount // 2
            mag = mag[..., crop_left:crop_left + max_frames]
            
        spectrograms.append(mag)
        
    # Stack into [N, 1, Freq, Time]
    X_stack = torch.stack(spectrograms)
    Y_stack = torch.tensor(labels, dtype=torch.long)
    
    # Normalize Log-Magnitudes strictly to [0, 1] range.
    # This is crucial for NMF to properly anchor the "silence" to 0.
    X_min = X_stack.amin(dim=(1, 2, 3), keepdim=True)
    X_stack = X_stack - X_min
    
    if use_instance_norm:
        X_max = X_stack.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    else:
        X_max = X_stack.max().clamp(min=1e-8)
        
    X_stack = X_stack / X_max

    # Return as [N, Freq, Time]. 
    # This allows ConvNMF to treat frequency bins as channels 
    # and slide motifs over the temporal dimension.
    X_out = X_stack.squeeze(1)  # [N, Freq, max_frames]

    return TensorDataset(X_out, Y_stack)


def get_librimix_dataset(
    root_dir: str = "./data/Libri2Mix",
    split: str = "test",
    sample_rate: int = 8000,
    mix_type: str = "min",
    mixture_dirname: str = "mix_clean",
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int | None = None,
    center: bool = True,
    max_frames: int | None = None,
    crop_mode: str = "center",
):
    """Return Libri2Mix dataset with mixture/source spectrogram triplets."""
    from data.librimix_dataset import LibriMixDataset

    return LibriMixDataset(
        root_dir=root_dir,
        split=split,
        sample_rate=sample_rate,
        mix_type=mix_type,
        mixture_dirname=mixture_dirname,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        max_frames=max_frames,
        crop_mode=crop_mode,
    )


def get_librimix_dataloader(
    root_dir: str = "./data/Libri2Mix",
    split: str = "test",
    sample_rate: int = 8000,
    mix_type: str = "min",
    mixture_dirname: str = "mix_clean",
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int | None = None,
    center: bool = True,
    max_frames: int | None = None,
    crop_mode: str = "center",
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Return Libri2Mix DataLoader with variable-length padding collate."""
    from data.librimix_dataset import get_librimix_dataloader as _loader

    return _loader(
        root_dir=root_dir,
        split=split,
        sample_rate=sample_rate,
        mix_type=mix_type,
        mixture_dirname=mixture_dirname,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        max_frames=max_frames,
        crop_mode=crop_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
