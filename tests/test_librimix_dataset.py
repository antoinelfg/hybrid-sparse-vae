from pathlib import Path
import sys

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.librimix_dataset import LibriMixDataset, get_librimix_dataloader


def _write_wav(path: Path, sr: int, data: np.ndarray) -> None:
    data_i16 = np.clip(data, -1.0, 1.0)
    data_i16 = (data_i16 * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, data_i16)


def test_librimix_dataset_triplet_and_phase(tmp_path):
    sr = 8000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    s1 = 0.4 * np.sin(2 * np.pi * 220 * t)
    s2 = 0.3 * np.sin(2 * np.pi * 440 * t)
    mix = s1 + s2

    split_dir = tmp_path / "wav8k" / "min" / "test"
    for sub in ["mix_clean", "s1", "s2"]:
        (split_dir / sub).mkdir(parents=True, exist_ok=True)

    _write_wav(split_dir / "mix_clean" / "utt0.wav", sr, mix)
    _write_wav(split_dir / "s1" / "utt0.wav", sr, s1)
    _write_wav(split_dir / "s2" / "utt0.wav", sr, s2)

    ds = LibriMixDataset(
        root_dir=str(tmp_path),
        split="test",
        sample_rate=sr,
        mix_type="min",
        mixture_dirname="mix_clean",
        n_fft=256,
        hop_length=64,
    )
    item = ds[0]

    assert item["mixture_mag"].shape == item["source1_mag"].shape == item["source2_mag"].shape
    assert torch.is_complex(item["mixture_complex"])
    assert item["mixture_complex"].shape == item["mixture_mag"].shape
    assert item["length"] == sr
    assert item["utt_id"] == "utt0"

    loader = get_librimix_dataloader(
        root_dir=str(tmp_path),
        split="test",
        sample_rate=sr,
        mix_type="min",
        mixture_dirname="mix_clean",
        n_fft=256,
        hop_length=64,
        batch_size=1,
    )
    batch = next(iter(loader))
    assert batch["mixture_mag"].shape[0] == 1
    assert torch.is_complex(batch["mixture_complex"])


def test_librimix_dataset_random_crop_changes_excerpt(tmp_path):
    sr = 8000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
    ramp = np.linspace(0.2, 1.0, t.shape[0], dtype=np.float32)
    s1 = ramp * np.sin(2 * np.pi * (180 + 120 * t) * t)
    s2 = (1.0 - 0.5 * ramp) * np.sin(2 * np.pi * (320 + 80 * t) * t)
    mix = s1 + s2

    split_dir = tmp_path / "wav8k" / "min" / "train-100"
    for sub in ["mix_clean", "s1", "s2"]:
        (split_dir / sub).mkdir(parents=True, exist_ok=True)

    _write_wav(split_dir / "mix_clean" / "utt0.wav", sr, mix)
    _write_wav(split_dir / "s1" / "utt0.wav", sr, s1)
    _write_wav(split_dir / "s2" / "utt0.wav", sr, s2)

    ds = LibriMixDataset(
        root_dir=str(tmp_path),
        split="train-100",
        sample_rate=sr,
        mix_type="min",
        mixture_dirname="mix_clean",
        n_fft=256,
        hop_length=64,
        max_frames=32,
        crop_mode="random",
    )

    crops = [ds[0]["mixture_mag"] for _ in range(6)]
    assert all(crop.shape[-1] == 32 for crop in crops)
    assert any(not torch.allclose(crops[0], crop) for crop in crops[1:])
