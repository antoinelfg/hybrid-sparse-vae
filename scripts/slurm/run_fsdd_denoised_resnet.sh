#!/bin/bash
#SBATCH --job-name=fsdd_denoised_resnet
#SBATCH --output=fsdd_denoised_resnet_%j.out
#SBATCH --error=fsdd_denoised_resnet_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

# Changes vs fsdd_constrained_resnet:
#   k_max: 1.5 → 0.8     (enforce sub-unit Gamma shape = sparser, more impulsive activations)
#   motif_width: 16 → 32  (richer atoms with more temporal context)
#   denoise: True          (Wiener-style noise-floor subtraction before log-compression)

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
    save_dir="./checkpoints/fsdd_denoised_resnet"
