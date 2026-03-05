#!/usr/bin/env python
"""Quick single-sample smoke test for hybrid-affinity BSS inference.

Loads one Libri2Mix sample, runs atom clustering + Wiener masking, and saves
`mix/s1_est/s2_est` wav files for listening.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick test for scripts/inference_bss.py")
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
    parser.add_argument("--output-dir", type=str, default="results/test_inference_bss")

    # Optional fallback model config if no hydra config is near checkpoint.
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

    script = Path(__file__).resolve().parent / "inference_bss.py"
    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        args.checkpoint,
        "--librimix-root",
        args.librimix_root,
        "--split",
        args.split,
        "--mix-type",
        args.mix_type,
        "--mixture-dirname",
        args.mixture_dirname,
        "--sample-index",
        str(args.sample_index),
        "--sample-rate",
        str(args.sample_rate),
        "--n-fft",
        str(args.n_fft),
        "--hop-length",
        str(args.hop_length),
        "--win-length",
        str(args.win_length),
        "--alpha",
        str(args.alpha),
        "--mask-power",
        str(args.mask_power),
        "--h-representation",
        args.h_representation,
        "--device",
        args.device,
        "--output-dir",
        args.output_dir,
        "--input-channels",
        str(args.input_channels),
        "--input-length",
        str(args.input_length),
        "--encoder-type",
        args.encoder_type,
        "--decoder-type",
        args.decoder_type,
        "--encoder-output-dim",
        str(args.encoder_output_dim),
        "--n-atoms",
        str(args.n_atoms),
        "--latent-dim",
        str(args.latent_dim),
        "--dict-init",
        args.dict_init,
        "--k-min",
        str(args.k_min),
        "--k-max",
        str(args.k_max),
        "--magnitude-dist",
        args.magnitude_dist,
        "--structure-mode",
        args.structure_mode,
        "--motif-width",
        str(args.motif_width),
        "--decoder-stride",
        str(args.decoder_stride),
        "--save-affinity",
    ]

    print("Running quick BSS inference test...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone. Audio outputs saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
