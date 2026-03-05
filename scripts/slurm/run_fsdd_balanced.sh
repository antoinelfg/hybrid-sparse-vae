#!/bin/bash
#SBATCH --job-name=fsdd_balanced
#SBATCH --output=fsdd_balanced_%j.out
#SBATCH --error=fsdd_balanced_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

# === fsdd_balanced: The Sweet Spot ===
# 
# Goal: Keep the clean silence of MaskedMSE but restore the rich timbre.
#
# Changes vs Final Push:
# 1. lambda_recon_l1 = 0.0 (Removed: it was filtering the harmonics too much)
# 2. lambda_silence_final = 0.3 (Reduced from 0.5: less aggressive on the mask boundaries)
# 3. beta_delta_final = 0.1 (Increased from 0.05: better atom competition)

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
    lambda_silence_final=0.3 \
    lambda_recon_l1=0.0 \
    beta_delta_final=0.1 \
    save_dir="./checkpoints/fsdd_balanced"
