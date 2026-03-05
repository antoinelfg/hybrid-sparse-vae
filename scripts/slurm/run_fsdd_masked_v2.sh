#!/bin/bash
#SBATCH --job-name=fsdd_masked_v2
#SBATCH --output=fsdd_masked_v2_%j.out
#SBATCH --error=fsdd_masked_v2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

# Fix vs fsdd_masked_resnet (job 4718855):
#   COLLAPSE ROOT CAUSE: masked signal_loss was normalised by n_signal pixels
#   → O(0.01), vs KL terms O(1-10), making all-δ=0 a local minimum.
#   FIX: Both signal_loss and silence_loss now normalised by batch_size
#   (same scale as KL), so the gradient cannot be overridden by KL pressure.
#
#   lambda_silence: 0.1 → 0.05  (lighter silence pressure to avoid competing
#                                 with signal gradient early in training)
#   beta_delta_final: default → 0.05  (less pressure to deactivate atoms
#                                      during Phase 3/4 ramp)

PYTHONPATH=. python train.py \
    dataset="fsdd" \
    decoder_type="convnmf" \
    encoder_type="resnet" \
    motif_width=32 \
    n_atoms=128 \
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
    beta_delta_final=0.05 \
    save_dir="./checkpoints/fsdd_masked_v2"
