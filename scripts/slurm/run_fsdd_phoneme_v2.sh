#!/bin/bash
#SBATCH --job-name=fsdd_phoneme_v2
#SBATCH --output=fsdd_phoneme_v2_%j.out
#SBATCH --error=fsdd_phoneme_v2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

# === fsdd_phoneme_v2: Overlapping Phonemes ===
#
# Fix for v1 halos:
#   decoder_stride: 8 -> 4 (50% overlap with motif_width=8)
#   This allows smooth Overlap-Add across segments.
#
# Other improvements:
#   lambda_recon_l1: 0.0 (Following Balanced run success)
#   lambda_silence_final: 0.3 (Slightly less aggressive)

PYTHONPATH=. python train.py \
    dataset="fsdd" \
    decoder_type="convnmf" \
    encoder_type="lista" \
    motif_width=8 \
    decoder_stride=4 \
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
    lambda_silence_final=0.3 \
    lambda_recon_l1=0.0 \
    beta_delta_final=0.1 \
    save_dir="./checkpoints/fsdd_phoneme_v2"
