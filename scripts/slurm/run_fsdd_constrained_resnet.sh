#!/bin/bash
#SBATCH --job-name=fsdd_constrained_resnet
#SBATCH --output=fsdd_constrained_resnet_%j.out
#SBATCH --error=fsdd_constrained_resnet_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

PYTHONPATH=. python train.py \
    dataset="fsdd" \
    decoder_type="convnmf" \
    encoder_type="resnet" \
    motif_width=16 \
    n_atoms=128 \
    latent_dim=64 \
    epochs=2000 \
    phase1_end=400 \
    phase2_end=500 \
    phase3_end=1000 \
    k_max=1.5 \
    delta_prior="'0.01,0.98,0.01'" \
    save_dir="./checkpoints/fsdd_constrained_resnet"
