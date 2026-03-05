#!/bin/bash
#SBATCH --job-name=fsdd_phoneme
#SBATCH --output=fsdd_phoneme_%j.out
#SBATCH --error=fsdd_phoneme_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

# === fsdd_phoneme: Phoneme-Scale Atoms (motif_width=8) ===
#
# vs fsdd_masked_v2 (motif_width=32):
#
#   motif_width: 32 → 8
#     Each atom now spans ~0.13s (phoneme) vs ~0.5s (word).
#     Target: atoms should learn harmonic bars (vowel formants)
#     rather than full-utterance energy blobs.
#
#   decoder_stride: 16 → 8 (= motif_width, no temporal gap)
#     With stride=8 and max_frames=64: T_latent = 64//8 = 8 steps.
#     ConvTranspose1d output = (8-1)*8 + 8 = 64 frames ✓
#
#   n_atoms: 128 → 256
#     Richer phoneme vocabulary. With k_max=0.8 and 8 T_latent steps,
#     we expect ~6-8 distinct atoms active per utterance.
#
# Denoising + masked_recon + progressive silence pruning kept identical.

PYTHONPATH=. python train.py \
    dataset="fsdd" \
    decoder_type="convnmf" \
    encoder_type="lista" \
    motif_width=8 \
    decoder_stride=8 \
    n_atoms=256 \
    latent_dim=64 \
    epochs=2000 \
    phase1_end=400 \
    phase2_end=500 \
    phase3_end=1000 \
    k_max=0.8 \
    delta_prior="'0.01,0.98,0.01'" \
    denoise=True \
    masked_recon=True \
    lambda_silence=0.05 \
    lambda_silence_final=0.5 \
    lambda_recon_l1=0.002 \
    beta_delta_final=0.05 \
    save_dir="./checkpoints/fsdd_phoneme"
