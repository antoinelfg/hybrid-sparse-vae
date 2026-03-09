#!/usr/bin/env python
"""Batch renderer for paper media across the current experiment set."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"Skipping missing media: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render atoms, strokes, and spectrogram media for paper experiments")
    parser.add_argument(
        "--groups",
        type=str,
        default="all",
        help="Comma-separated subset of groups: sinusoid,mnist,fsdd,librimix_current,librimix_legacy,all",
    )
    parser.add_argument("--output-root", type=str, default="report/figures/experiment_media")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = {item.strip() for item in args.groups.split(",") if item.strip()}
    if "all" in selected:
        selected = {"sinusoid", "mnist", "fsdd", "librimix_current", "librimix_legacy"}

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    sinusoid_entries = [
        ("lowk_oldkl", REPO_ROOT / "results/regime_study/HSVAE-lowk-oldkl/seed_42/hybrid_vae_final.pt"),
        ("lowk_full", REPO_ROOT / "results/regime_study/HSVAE-lowk-full/seed_42/hybrid_vae_final.pt"),
        ("no_delta", REPO_ROOT / "results/regime_study/HSVAE-no-delta/seed_42/hybrid_vae_final.pt"),
    ]
    if "sinusoid" in selected:
        for name, ckpt in sinusoid_entries:
            if not ckpt.exists():
                print(f"Skipping missing sinusoid checkpoint: {ckpt}")
                continue
            out_dir = output_root / "sinusoid" / name
            if out_dir.exists() and not args.overwrite:
                print(f"Skipping existing directory: {out_dir}")
                continue
            _run([py, "scripts/vis_sinusoid_atoms.py", "--checkpoint", str(ckpt), "--output-dir", str(out_dir), "--device", args.device])

    mnist_entries = [
        ("lowk", REPO_ROOT / "results/mnist_regime/lowk_seed_42/hybrid_vae_final.pt"),
        ("highk", REPO_ROOT / "results/mnist_regime/highk_seed_42/hybrid_vae_final.pt"),
    ]
    if "mnist" in selected:
        for name, ckpt in mnist_entries:
            if not ckpt.exists():
                print(f"Skipping missing MNIST checkpoint: {ckpt}")
                continue
            out_dir = output_root / "mnist" / name
            if out_dir.exists() and not args.overwrite:
                print(f"Skipping existing directory: {out_dir}")
                continue
            _run([py, "scripts/vis_mnist_atoms.py", "--checkpoint", str(ckpt), "--output-dir", str(out_dir), "--device", args.device])

    fsdd_entries = [
        ("binary_additive_safe", REPO_ROOT / "checkpoints/fsdd_binary_additive_safe/hybrid_vae_final.pt"),
        ("ternary_signed_safe", REPO_ROOT / "checkpoints/fsdd_ternary_signed_safe/hybrid_vae_final.pt"),
    ]
    if "fsdd" in selected:
        for name, ckpt in fsdd_entries:
            if not ckpt.exists():
                print(f"Skipping missing FSDD checkpoint: {ckpt}")
                continue
            out_dir = output_root / "fsdd" / name
            if out_dir.exists() and not args.overwrite:
                print(f"Skipping existing directory: {out_dir}")
                continue
            _run([py, "scripts/vis_spectrogram_atoms.py", "--checkpoint", str(ckpt), "--output-dir", str(out_dir), "--device", args.device])
            _run([py, "scripts/vis_fsdd_recons.py", "--checkpoint", str(ckpt), "--output-dir", str(out_dir), "--device", args.device])

    librimix_entries = [
        ("direct_mask", REPO_ROOT / "checkpoints/librimix_direct_mask_baseline_e120"),
        ("hybrid_partition_klsafe", REPO_ROOT / "checkpoints/librimix_hybrid_partition_klsafe_n512_w16_s4_e200"),
    ]
    if "librimix_current" in selected:
        for name, ckpt in librimix_entries:
            if not ckpt.exists():
                print(f"Skipping missing LibriMix checkpoint dir: {ckpt}")
                continue
            out_dir = output_root / "librimix_current" / name
            if out_dir.exists() and not args.overwrite:
                print(f"Skipping existing directory: {out_dir}")
                continue
            _run([py, "scripts/vis_librimix_experiment.py", "--checkpoint", str(ckpt), "--output-dir", str(out_dir), "--device", args.device])

    legacy_entries = [
        ("v1_sample0", REPO_ROOT / "results/inference_bss/librimix_klt1_pure_200_sample0"),
        ("v2_sample0", REPO_ROOT / "results/inference_bss/librimix_v2_klt1_n512_w16_s4_e200_specs/sample_0"),
        ("v3_sample0", REPO_ROOT / "results/inference_bss/librimix_v3_archsafe_n512_w16_s4_e200_specs/sample_0"),
    ]
    if "librimix_legacy" in selected:
        for name, inf_dir in legacy_entries:
            if not inf_dir.exists():
                print(f"Skipping missing legacy inference dir: {inf_dir}")
                continue
            existing = next(inf_dir.glob("*_spectrograms.png"), None)
            if existing is None:
                _run([py, "scripts/plot_bss_spectrograms.py", "--inference-dir", str(inf_dir)])
                existing = next(inf_dir.glob("*_spectrograms.png"), None)
            if existing is None:
                print(f"Skipping legacy dir without spectrogram panel: {inf_dir}")
                continue
            _copy_if_exists(existing, output_root / "librimix_legacy" / name / "spectrogram_panel.png")


if __name__ == "__main__":
    main()
