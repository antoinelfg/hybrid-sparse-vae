#!/bin/bash
#SBATCH --job-name=fsdd_final_push
#SBATCH --output=fsdd_final_push_%j.out
#SBATCH --error=fsdd_final_push_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

# === fsdd_final_push: Sparsity-Driven Silence Pruning ===
#
# Two new mechanisms vs masked_v2:
#
# 1. PROGRESSIVE lambda_silence (0.05 → 0.5):
#    - P1/P2: lambda=0.05, model learns to hit signal peaks freely
#    - P3/P4: lambda ramps to 0.5, model becomes "terrified" of silence
#    - Mirrors the beta-KL schedule so sparsity pressure matches structure pressure
#
# 2. L1 NOISE-FLOOR PENALTY (lambda_recon_l1=0.002):
#    - Constant pull toward 0 on every output pixel
#    - Forces decoder to "earn" every non-zero pixel against signal_loss gradient
#    - L1 (not L2) induces true sparsity: flat gradient near 0, no saturation
#    - Small enough (0.002) not to compete with signal_loss (~50), just prune fog

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
    lambda_silence_final=0.5 \
    lambda_recon_l1=0.002 \
    beta_delta_final=0.05 \
    save_dir="./checkpoints/fsdd_final_push"
